// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// RDNA3/4 register-array GDN Q8: S_local[128] lives in registers.
// 128 VGPRs for S + ~16 compute = 144 total. 1536/144 = 10 waves/SIMD.
// S never touches memory during recurrence — pure ALU.
// Dequant at start, requant at end. Inner loop is register-only.

#define HD 128

extern "C" __launch_bounds__(128, 10)
__global__ void gated_delta_net_q8(
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
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int row = threadIdx.x;  // 0..127, one thread per S row
    const int stride = n_heads * HD;

    // Dequant S row into registers — stays here for entire recurrence
    signed char* S_q8_row = s_q8 + h * HD * HD + row * HD;
    float row_scale = s_scales[h * HD + row];
    float S_local[HD];
    for (int j = 0; j < HD; j++)
        S_local[j] = row_scale * (float)S_q8_row[j];

    // k and q shared via LDS (128 threads load cooperatively)
    __shared__ float k_lds[HD];
    __shared__ float q_lds[HD];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        k_lds[row] = k_t[row];
        q_lds[row] = q_t[row];
        __syncthreads();

        // kv = S_row @ k — dot product from registers
        float kv = 0.0f;
        for (int j = 0; j < HD; j += 4) {
            kv += S_local[j]   * k_lds[j];
            kv += S_local[j+1] * k_lds[j+1];
            kv += S_local[j+2] * k_lds[j+2];
            kv += S_local[j+3] * k_lds[j+3];
        }

        float delta = (v_t[row] - alpha * kv) * beta_v;

        // Update S in registers + compute output dot
        float out_val = 0.0f;
        for (int j = 0; j < HD; j += 4) {
            float s0 = alpha * S_local[j]   + k_lds[j]   * delta;
            float s1 = alpha * S_local[j+1] + k_lds[j+1] * delta;
            float s2 = alpha * S_local[j+2] + k_lds[j+2] * delta;
            float s3 = alpha * S_local[j+3] + k_lds[j+3] * delta;
            S_local[j]   = s0;
            S_local[j+1] = s1;
            S_local[j+2] = s2;
            S_local[j+3] = s3;
            out_val += s0 * q_lds[j] + s1 * q_lds[j+1] + s2 * q_lds[j+2] + s3 * q_lds[j+3];
        }
        output[t * stride + h * HD + row] = out_val;
        __syncthreads();
    }

    // Requant registers → Q8 global
    float amax = 0.0f;
    for (int j = 0; j < HD; j++) {
        float a = fabsf(S_local[j]);
        if (a > amax) amax = a;
    }
    float new_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    for (int j = 0; j < HD; j++) {
        int val = (int)roundf(S_local[j] * inv_scale);
        S_q8_row[j] = (signed char)(val < -128 ? -128 : (val > 127 ? 127 : val));
    }
    s_scales[h * HD + row] = new_scale;
}
