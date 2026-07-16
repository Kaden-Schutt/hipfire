// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_CQN_BLOCK
#define HIPFIRE_CQN_BLOCK 128
#endif

#ifndef HIPFIRE_CQN_KERNEL
#define HIPFIRE_CQN_KERNEL conv1d_silu_split_qknorm_b128
#endif

#define HD 128
#if HIPFIRE_CQN_BLOCK >= 256
#define HEADS_PER_WG (HIPFIRE_CQN_BLOCK / 256)
#else
#define HEADS_PER_WG 1
#endif

// gfx1100/gfx1201 decode-only fusion of conv1d_silu_split_f32 and
// fused_qk_l2_norm_scale_f32. Q/K conv results stage in LDS, then the first
// wave for each head performs the exact existing four-value lane sum, XOR
// reduction, and two-step Q scale. V channels retain the original independent
// conv path. Q/K reach global memory only once, already normalized.
extern "C" __launch_bounds__(HIPFIRE_CQN_BLOCK)
__global__ void HIPFIRE_CQN_KERNEL(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
#if defined(HIPFIRE_CQN_SCALAR_PREP)
    float* __restrict__ beta,
    float* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
#endif
    int k_dim,
    int v_dim,
    int n_heads,
    int head_dim,
    float q_scale,
    float eps
#if defined(HIPFIRE_CQN_SCALAR_PREP)
    , int n_v_heads
#endif
) {
    __shared__ float q_s[HEADS_PER_WG * HD];
    __shared__ float k_s[HEADS_PER_WG * HD];

    const int tid = threadIdx.x;
    const int qk_blocks = (n_heads + HEADS_PER_WG - 1) / HEADS_PER_WG;

#if defined(HIPFIRE_CQN_SCALAR_PREP)
    // The scalar preparation is independent of the conv/QK-normalization
    // work. Give it one extra workgroup in this already-required launch; the
    // launch boundary then orders both products before the GDN consumer.
    const int v_blocks = (v_dim + HIPFIRE_CQN_BLOCK - 1) / HIPFIRE_CQN_BLOCK;
    if ((int)blockIdx.x == qk_blocks + v_blocks) {
        if (tid < n_v_heads) {
            beta[tid] = 1.0f / (1.0f + expf(-beta[tid]));
            const float biased = alpha[tid] + dt_bias[tid];
            const float sp = (biased > 20.0f)
                ? biased
                : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
            alpha[tid] = sp * (-expf(a_log[tid]));
        }
        return;
    }
#endif

    if ((int)blockIdx.x < qk_blocks) {
#if HIPFIRE_CQN_BLOCK >= 256
        const int local_head = tid / 256;
        const int local = tid - local_head * 256;
        const int h = blockIdx.x * HEADS_PER_WG + local_head;
        if (h < n_heads) {
            const bool is_k = local >= HD;
            const int d = local & (HD - 1);
            const int c = is_k ? (k_dim + h * HD + d) : (h * HD + d);

            // Verbatim conv1d_silu_split_f32 math and state-update order.
            float x = input[c];
            float s0 = state[c * 3];
            float s1 = state[c * 3 + 1];
            float s2 = state[c * 3 + 2];
            float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                    + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
            float result = y / (1.0f + expf(-y));
            state[c * 3 + 2] = s1;
            state[c * 3 + 1] = s0;
            state[c * 3] = x;
            if (is_k) k_s[local_head * HD + d] = result;
            else      q_s[local_head * HD + d] = result;
        }
#else
        const int h = blockIdx.x;
        for (int d = tid; d < HD; d += HIPFIRE_CQN_BLOCK) {
            const int qc = h * HD + d;
            const int kc = k_dim + h * HD + d;

            float qx = input[qc];
            float qs0 = state[qc * 3];
            float qs1 = state[qc * 3 + 1];
            float qs2 = state[qc * 3 + 2];
            float qy = weight[qc * 4 + 3] * qx + weight[qc * 4 + 2] * qs0
                     + weight[qc * 4 + 1] * qs1 + weight[qc * 4] * qs2;
            float qr = qy / (1.0f + expf(-qy));
            state[qc * 3 + 2] = qs1;
            state[qc * 3 + 1] = qs0;
            state[qc * 3] = qx;
            q_s[d] = qr;

            float kx = input[kc];
            float ks0 = state[kc * 3];
            float ks1 = state[kc * 3 + 1];
            float ks2 = state[kc * 3 + 2];
            float ky = weight[kc * 4 + 3] * kx + weight[kc * 4 + 2] * ks0
                     + weight[kc * 4 + 1] * ks1 + weight[kc * 4] * ks2;
            float kr = ky / (1.0f + expf(-ky));
            state[kc * 3 + 2] = ks1;
            state[kc * 3 + 1] = ks0;
            state[kc * 3] = kx;
            k_s[d] = kr;
        }
#endif
        __syncthreads();

#if HIPFIRE_CQN_BLOCK >= 256
        const int reduce_head = tid / 256;
        const int lane = tid & 255;
        const int reduce_h = blockIdx.x * HEADS_PER_WG + reduce_head;
        if (reduce_h < n_heads && lane < 32) {
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int d = lane; d < HD; d += 32) {
                const float qv = q_s[reduce_head * HD + d];
                const float kv = k_s[reduce_head * HD + d];
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
            for (int d = lane; d < HD; d += 32) {
                float qv = q_s[reduce_head * HD + d];
                qv *= q_inv;
                qv *= q_scale;
                q_out[reduce_h * HD + d] = qv;
                k_out[reduce_h * HD + d] = k_s[reduce_head * HD + d] * k_inv;
            }
        }
#else
        if (tid < 32) {
            const int h = blockIdx.x;
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int d = tid; d < HD; d += 32) {
                const float qv = q_s[d];
                const float kv = k_s[d];
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
            for (int d = tid; d < HD; d += 32) {
                float qv = q_s[d];
                qv *= q_inv;
                qv *= q_scale;
                q_out[h * HD + d] = qv;
                k_out[h * HD + d] = k_s[d] * k_inv;
            }
        }
#endif
        return;
    }

    const int v_block = blockIdx.x - qk_blocks;
    const int vi = v_block * HIPFIRE_CQN_BLOCK + tid;
    if (vi >= v_dim) return;
    const int c = 2 * k_dim + vi;

    // Verbatim V-channel path from conv1d_silu_split_f32.
    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];
    float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
    float result = y / (1.0f + expf(-y));
    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3] = x;
    v_out[vi] = result;
}
