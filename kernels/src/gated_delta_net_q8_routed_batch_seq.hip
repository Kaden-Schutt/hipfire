// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 128
#define TILE_ROWS 4

// Routed Q8 Gated Delta Net recurrence for independent request sessions.
//
// Grid: [n_heads, HD/TILE_ROWS, n_sessions]; block [32].
//
// Each block owns one session/head/S-row-tile. It scans the flattened
// round-major prefix rows and applies only rows whose row_session_indices[b]
// match blockIdx.z. This preserves per-session recurrence order while avoiding
// the invalid shared-state concatenation across independent requests.
extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8_routed_batch_seq(
    const float* __restrict__ q,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ k,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ v,                    // [batch_rows * n_heads * HD]
    const float* __restrict__ gate,                 // [batch_rows * n_heads]
    const float* __restrict__ beta,                 // [batch_rows * n_heads]
    const uint64_t* __restrict__ state_ptrs,        // [n_sessions * ptr_layer_stride] i8 S ptrs
    const uint64_t* __restrict__ scale_ptrs,        // [n_sessions * ptr_layer_stride] f32 scale ptrs
    const int* __restrict__ row_session_indices,    // [batch_rows]
    float* __restrict__ output,                     // [batch_rows * n_heads * HD]
    int ptr_layer_stride,
    int delta_layer_index,
    int batch_rows,
    int n_heads,
    int head_dim,
    unsigned int frame
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    const int session_idx = blockIdx.z;
    if (h >= n_heads || head_dim != HD) return;

    const int tid = threadIdx.x;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];

    signed char* s_q8 = reinterpret_cast<signed char*>(
        static_cast<uintptr_t>(state_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    float* s_scales = reinterpret_cast<float*>(
        static_cast<uintptr_t>(scale_ptrs[(size_t)session_idx * ptr_layer_stride + delta_layer_index])
    );
    signed char* sq_base = s_q8 + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32) {
        int r = i / HD;
        float scale = s_scales[h * HD + row_start + r];
        S_tile[i] = scale * (float)sq_base[i];
    }

    for (int b = 0; b < batch_rows; b++) {
        if (row_session_indices[b] != session_idx) continue;

        const float* q_t = q + b * stride + h * HD;
        const float* k_t = k + b * stride + h * HD;
        const float* v_t = v + b * stride + h * HD;
        float alpha = expf(gate[b * n_heads + h]);
        float beta_v = beta[b * n_heads + h];

        float k0 = k_t[tid], k1 = k_t[tid + 32], k2 = k_t[tid + 64], k3 = k_t[tid + 96];
        float q0 = q_t[tid], q1 = q_t[tid + 32], q2 = q_t[tid + 64], q3 = q_t[tid + 96];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD;
            float s0 = sr[tid], s1 = sr[tid + 32], s2 = sr[tid + 64], s3 = sr[tid + 96];

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
            sr[tid] = s0;
            sr[tid + 32] = s1;
            sr[tid + 64] = s2;
            sr[tid + 96] = s3;

            float out_v = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
            for (int o = 16; o > 0; o >>= 1) {
                out_v += __shfl_down(out_v, o);
            }
            if (tid == 0) {
                output[b * stride + h * HD + row] = out_v;
            }
        }
    }

    for (int r = 0; r < TILE_ROWS; r++) {
        float* sr = S_tile + r * HD;
        float s0 = sr[tid], s1 = sr[tid + 32], s2 = sr[tid + 64], s3 = sr[tid + 96];
        float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
        for (int o = 16; o > 0; o >>= 1) {
            my_max = fmaxf(my_max, __shfl_xor(my_max, o));
        }

        float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
        signed char* dst = sq_base + r * HD;
        unsigned int rng = (unsigned int)(h * 1664525u
                         + (row_start + r) * 1013904223u
                         + tid * 214013u
                         + frame * 2531011u);
        #define STOCH_ROUND(val) ({ \
            rng = rng * 1664525u + 1013904223u; \
            float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
            float scaled = (val) * inv_s; \
            (signed char)fminf(fmaxf(floorf(scaled + noise), -128.0f), 127.0f); \
        })
        dst[tid] = STOCH_ROUND(s0);
        dst[tid + 32] = STOCH_ROUND(s1);
        dst[tid + 64] = STOCH_ROUND(s2);
        dst[tid + 96] = STOCH_ROUND(s3);
        #undef STOCH_ROUND
        if (tid == 0) {
            s_scales[h * HD + row_start + r] = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
        }
    }
}
