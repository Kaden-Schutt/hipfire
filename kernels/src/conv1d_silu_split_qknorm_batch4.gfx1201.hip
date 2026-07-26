// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// N=2..4 verifier lift of the certified gfx1201 decode
// conv1d_silu_split_qknorm_b256 kernel.
//
// One 256-thread Q/K workgroup owns one head and advances all live verifier
// rows in causal order. This preserves the ordinary conv ring semantics while
// keeping the decode kernel's LDS staging and exact wave32 normalization
// arithmetic. V workgroups use the same one-thread-per-channel conv loop.
//
// Input:  [N, 2*k_dim + v_dim]
// Q/K:    [N, k_dim]
// V:      [N, v_dim]
// State:  [2*k_dim + v_dim, 3]
// Grid:   [n_heads + ceil(v_dim / 256), 1, 1]
// Block:  [256, 1, 1]

#define HD 128
#define BMAX 4

extern "C" __launch_bounds__(256)
__global__ void conv1d_silu_split_qknorm_batch4_gfx1201(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_heads,
    int head_dim,
    float q_scale,
    float eps,
    int n_tokens
) {
    if (head_dim != HD || n_tokens < 1 || n_tokens > BMAX) return;

    __shared__ float q_s[BMAX * HD];
    __shared__ float k_s[BMAX * HD];

    const int tid = threadIdx.x;
    const int qk_blocks = n_heads;
    const int n_channels = 2 * k_dim + v_dim;

    if ((int)blockIdx.x < qk_blocks) {
        const int h = blockIdx.x;
        const bool is_k = tid >= HD;
        const int d = tid & (HD - 1);
        const int c = is_k ? (k_dim + h * HD + d) : (h * HD + d);

        #pragma unroll
        for (int t = 0; t < BMAX; ++t) {
            if (t < n_tokens) {
                // Keep the load/expression/update order verbatim with
                // conv1d_silu_split_f32. The state round-trip between rows is
                // intentional: it is the causal verifier recurrence.
                const float x = input[(long long)t * n_channels + c];
                const float s0 = state[c * 3];
                const float s1 = state[c * 3 + 1];
                const float s2 = state[c * 3 + 2];
                const float y =
                    weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                    + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
                const float result = y / (1.0f + expf(-y));
                state[c * 3 + 2] = s1;
                state[c * 3 + 1] = s0;
                state[c * 3] = x;
                if (is_k) k_s[t * HD + d] = result;
                else      q_s[t * HD + d] = result;
            }
        }
        __syncthreads();

        if (tid < 32) {
            #pragma unroll
            for (int t = 0; t < BMAX; ++t) {
                if (t < n_tokens) {
                    float q_sq = 0.0f;
                    float k_sq = 0.0f;
                    #pragma unroll
                    for (int i = tid; i < HD; i += 32) {
                        const float qv = q_s[t * HD + i];
                        const float kv = k_s[t * HD + i];
                        q_sq += qv * qv;
                        k_sq += kv * kv;
                    }
                    for (int o = 16; o > 0; o >>= 1) {
                        q_sq += __shfl_xor(q_sq, o);
                        k_sq += __shfl_xor(k_sq, o);
                    }
                    const float q_inv = rsqrtf(q_sq + eps);
                    const float k_inv = rsqrtf(k_sq + eps);
                    #pragma unroll
                    for (int i = tid; i < HD; i += 32) {
                        float qv = q_s[t * HD + i];
                        qv *= q_inv;
                        qv *= q_scale;
                        q_out[(long long)t * k_dim + h * HD + i] = qv;
                        k_out[(long long)t * k_dim + h * HD + i] =
                            k_s[t * HD + i] * k_inv;
                    }
                }
            }
        }
        return;
    }

    const int vi = ((int)blockIdx.x - qk_blocks) * 256 + tid;
    if (vi >= v_dim) return;
    const int c = 2 * k_dim + vi;
    #pragma unroll
    for (int t = 0; t < BMAX; ++t) {
        if (t < n_tokens) {
            const float x = input[(long long)t * n_channels + c];
            const float s0 = state[c * 3];
            const float s1 = state[c * 3 + 1];
            const float s2 = state[c * 3 + 2];
            const float y =
                weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
            const float result = y / (1.0f + expf(-y));
            state[c * 3 + 2] = s1;
            state[c * 3 + 1] = s0;
            state[c * 3] = x;
            v_out[(long long)t * v_dim + vi] = result;
        }
    }
}
