// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <math.h>

// gfx1151 register-state GDN Q8 experiment.
//
// The generic kernel maps one wave32 to four S rows and stores those rows in
// LDS. This variant maps one thread to one S row, keeps the full row in
// registers through the recurrence, and launches one 128-thread block per
// head. It preserves the production ABI, including the host-provided stochastic
// requantization frame.

#if defined(__gfx1151__)

#define HD 128

extern "C" __launch_bounds__(128, 10)
__global__ void gated_delta_net_q8_reg_gfx1151(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame
) {
    const int h = blockIdx.x;
    if (h >= n_heads || head_dim != HD) return;

    const int row = threadIdx.x;
    const int stride = n_heads * HD;

    signed char* s_row = s_q8 + h * HD * HD + row * HD;
    const float row_scale = s_scales[h * HD + row];

    float S[HD];
    #pragma unroll
    for (int j = 0; j < HD; ++j) {
        S[j] = row_scale * (float)s_row[j];
    }

    __shared__ float k_lds[HD];
    __shared__ float q_lds[HD];

    for (int t = 0; t < n_tokens; ++t) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;

        k_lds[row] = k_t[row];
        q_lds[row] = q_t[row];
        __syncthreads();

        const float alpha = expf(gate[t * n_heads + h]);
        const float beta_v = beta[t * n_heads + h];

        float kv = 0.0f;
        #pragma unroll
        for (int j = 0; j < HD; j += 4) {
            kv += S[j + 0] * k_lds[j + 0];
            kv += S[j + 1] * k_lds[j + 1];
            kv += S[j + 2] * k_lds[j + 2];
            kv += S[j + 3] * k_lds[j + 3];
        }

        const float delta = (v_t[row] - alpha * kv) * beta_v;

        float out_v = 0.0f;
        #pragma unroll
        for (int j = 0; j < HD; j += 4) {
            float s0 = alpha * S[j + 0] + k_lds[j + 0] * delta;
            float s1 = alpha * S[j + 1] + k_lds[j + 1] * delta;
            float s2 = alpha * S[j + 2] + k_lds[j + 2] * delta;
            float s3 = alpha * S[j + 3] + k_lds[j + 3] * delta;
            S[j + 0] = s0;
            S[j + 1] = s1;
            S[j + 2] = s2;
            S[j + 3] = s3;
            out_v += s0 * q_lds[j + 0];
            out_v += s1 * q_lds[j + 1];
            out_v += s2 * q_lds[j + 2];
            out_v += s3 * q_lds[j + 3];
        }

        output[t * stride + h * HD + row] = out_v;
        __syncthreads();
    }

    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < HD; ++j) {
        amax = fmaxf(amax, fabsf(S[j]));
    }

    const float inv_s = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    unsigned int rng = (unsigned int)(h * 1664525u + row * 1013904223u
                     + frame * 2531011u);

    #pragma unroll
    for (int j = 0; j < HD; ++j) {
        rng = rng * 1664525u + 1013904223u + (unsigned int)(j * 214013u);
        const float noise = (float)(rng >> 8) * (1.0f / 16777216.0f);
        const float scaled = S[j] * inv_s;
        s_row[j] = (signed char)fminf(fmaxf(floorf(scaled + noise), -128.0f), 127.0f);
    }

    s_scales[h * HD + row] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
}

#else

extern "C" __global__ void gated_delta_net_q8_reg_gfx1151(
    const float*, const float*, const float*, const float*, const float*,
    signed char*, float*, float*, int, int, int, unsigned int
) {}

#endif
