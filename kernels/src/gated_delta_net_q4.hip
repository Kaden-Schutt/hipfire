// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#define HD 128

extern "C" __global__ void gated_delta_net_q4(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    unsigned char* __restrict__ s_q4,   // [n_heads × HD × HD/2] nibble-packed
    float* __restrict__ s_scales,       // [n_heads × HD] per-row scale
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int row = threadIdx.x;  // 0..127

    const int stride = n_heads * HD;
    const int half_hd = HD / 2;  // 64 bytes per row (nibble packed)
    unsigned char* S_q4_row = s_q4 + h * HD * half_hd + row * half_hd;
    float row_scale = s_scales[h * HD + row];

    // Dequantize S row from Q4 to FP32 in registers
    // Nibble pack: byte[j/2] = low_nibble(j even) | high_nibble(j odd)
    // Signed 4-bit: -8..+7 stored as unsigned 0..15, subtract 8
    float S_local[HD];
    for (int j = 0; j < HD; j += 2) {
        unsigned char byte = S_q4_row[j / 2];
        S_local[j]   = row_scale * (float)((int)(byte & 0xF) - 8);
        S_local[j+1] = row_scale * (float)((int)(byte >> 4) - 8);
    }

    __shared__ float k_lds[HD];
    __shared__ float q_lds[HD];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha_val = expf(gate[t * n_heads + h]);
        float beta_val = beta[t * n_heads + h];

        k_lds[row] = k_t[row];
        q_lds[row] = q_t[row];
        __syncthreads();

        // Phase 1: kv = (S @ k)[row] — row dot product (NOT S^T @ k!)
        float kv_val = 0.0f;
        for (int j = 0; j < HD; j += 4) {
            kv_val += S_local[j]   * k_lds[j];
            kv_val += S_local[j+1] * k_lds[j+1];
            kv_val += S_local[j+2] * k_lds[j+2];
            kv_val += S_local[j+3] * k_lds[j+3];
        }

        float delta = (v_t[row] - alpha_val * kv_val) * beta_val;

        // Phase 2+3 fused: update S in registers and compute output
        float out_val = 0.0f;
        for (int j = 0; j < HD; j += 4) {
            float s0 = alpha_val * S_local[j]   + k_lds[j]   * delta;
            float s1 = alpha_val * S_local[j+1] + k_lds[j+1] * delta;
            float s2 = alpha_val * S_local[j+2] + k_lds[j+2] * delta;
            float s3 = alpha_val * S_local[j+3] + k_lds[j+3] * delta;
            S_local[j]   = s0;
            S_local[j+1] = s1;
            S_local[j+2] = s2;
            S_local[j+3] = s3;
            out_val += s0 * q_lds[j] + s1 * q_lds[j+1] + s2 * q_lds[j+2] + s3 * q_lds[j+3];
        }
        output[t * stride + h * HD + row] = out_val;

        // Per-token Q4 requant roundtrip — matches the per-token decode cadence
        // (gated_delta_net_q4 with n_tokens=1 requants S after every token).
        // Quantize S_local → Q4 and dequant back in registers so the next token
        // reads the Q4-roundtripped state; global store on the last token only.
        // Deterministic stochastic rounding with a state-derived seed (no
        // token-index term) makes this N-token launch bit-exact with N×1 decode.
        {
            // Quantize S row back to Q4: per-row absmax, symmetric -8..+7
            float amax = 0.0f;
            for (int j = 0; j < HD; j++) {
                float a = fabsf(S_local[j]);
                if (a > amax) amax = a;
            }
            float new_scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
            float inv_scale = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
            bool last_tok = (t == n_tokens - 1);

            // Stochastic rounding: eliminates systematic bias that crushes small state values
            unsigned int rng = (unsigned int)(h * 1664525u + row * 1013904223u + __float_as_uint(amax) * 2531011u);
            for (int j = 0; j < HD; j += 2) {
                rng = rng * 1664525u + 1013904223u;
                float n0 = (float)(rng >> 8) * (1.0f / 16777216.0f);
                rng = rng * 1664525u + 1013904223u;
                float n1 = (float)(rng >> 8) * (1.0f / 16777216.0f);
                int v0 = (int)floorf(S_local[j] * inv_scale + n0) + 8;
                int v1 = (int)floorf(S_local[j+1] * inv_scale + n1) + 8;
                v0 = v0 < 0 ? 0 : (v0 > 15 ? 15 : v0);
                v1 = v1 < 0 ? 0 : (v1 > 15 ? 15 : v1);
                // Dequant back so the next token reads the Q4 state.
                S_local[j]   = new_scale * (float)(v0 - 8);
                S_local[j+1] = new_scale * (float)(v1 - 8);
                if (last_tok) S_q4_row[j / 2] = (unsigned char)(v0 | (v1 << 4));
            }
            if (last_tok) s_scales[h * HD + row] = new_scale;
        }
        __syncthreads();
    }
}
