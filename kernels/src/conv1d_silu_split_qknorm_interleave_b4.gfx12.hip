// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128
#define BATCH 4
#define BLOCK 256

// gfx1201 MTP verifier fusion for the fixed Qwen3.6 B=4 LA shape.
//
// Q/K workgroups advance all four conv states in token order, retain the
// results in LDS, then perform the same wave32 reductions and two-step Q
// scaling as fused_qk_l2_norm_scale_interleave_f32_batched. The normalized
// values are written directly into the expanded value-head layout. V
// workgroups retain the independent channel loop from conv1d_silu_split_f32.
//
// This replaces:
//   conv1d_silu_split_f32_n
//   fused_qk_l2_norm_scale_interleave_f32_batched
// with one dispatch without changing the recurrent state trajectory.
extern "C" __launch_bounds__(BLOCK)
__global__ void conv1d_silu_split_qknorm_interleave_b4(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_heads,
    int ratio,
    int head_dim,
    int n_tokens,
    float q_scale,
    float eps
) {
    __shared__ float q_s[BATCH][HD];
    __shared__ float k_s[BATCH][HD];

    const int tid = threadIdx.x;
    const int n_channels = 2 * k_dim + v_dim;

    if ((int)blockIdx.x < n_heads) {
        const int h = blockIdx.x;
        const bool is_k = tid >= HD;
        const int d = tid & (HD - 1);
        const int c = (is_k ? k_dim : 0) + h * HD + d;

        #pragma unroll
        for (int t = 0; t < BATCH; t++) {
            // Preserve the baseline conv expression and load placement. The
            // state is advanced in exactly the same token order per channel.
            float x = input[t * n_channels + c];
            float s0 = state[c * 3];
            float s1 = state[c * 3 + 1];
            float s2 = state[c * 3 + 2];
            float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                    + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
            float result = y / (1.0f + expf(-y));
            state[c * 3 + 2] = s1;
            state[c * 3 + 1] = s0;
            state[c * 3] = x;
            if (is_k) k_s[t][d] = result;
            else      q_s[t][d] = result;
        }
        __syncthreads();

        if (tid < 32) {
            const int n_value_heads = n_heads * ratio;
            #pragma unroll
            for (int t = 0; t < BATCH; t++) {
                float q_sq = 0.0f;
                float k_sq = 0.0f;
                #pragma unroll
                for (int i = tid; i < HD; i += 32) {
                    const float qv = q_s[t][i];
                    const float kv = k_s[t][i];
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
                for (int r = 0; r < ratio; r++) {
                    const int vh = h * ratio + r;
                    float* q_dst = q_out + ((long long)t * n_value_heads + vh) * HD;
                    float* k_dst = k_out + ((long long)t * n_value_heads + vh) * HD;
                    #pragma unroll
                    for (int i = tid; i < HD; i += 32) {
                        float qv = q_s[t][i] * q_inv;
                        qv *= q_scale;
                        q_dst[i] = qv;
                        k_dst[i] = k_s[t][i] * k_inv;
                    }
                }
            }
        }
        return;
    }

    const int vi = ((int)blockIdx.x - n_heads) * BLOCK + tid;
    if (vi >= v_dim) return;
    const int c = 2 * k_dim + vi;
    #pragma unroll
    for (int t = 0; t < BATCH; t++) {
        float x = input[t * n_channels + c];
        float s0 = state[c * 3];
        float s1 = state[c * 3 + 1];
        float s2 = state[c * 3 + 2];
        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
        float result = y / (1.0f + expf(-y));
        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3] = x;
        v_out[t * v_dim + vi] = result;
    }
}
