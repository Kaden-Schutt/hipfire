// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on an asym4 post-RoPE K cache.
//
// asym4 stores K as: [4B cnorm][head_dim/2 packed nibbles] with the same
// per-band Givens rotation asym3 uses but a 4-bit (16-centroid) codebook.
// The write-time layout runs in `n_halves = head_dim/128` passes, each
// thread owning 4 consecutive dims per pass; cos_theta / sin_theta are
// indexed by `block_tid = half * 32 + tid` with `block_tid * 2 + {0,1}`
// as the band index. Matches kv_cache_write_asym_k_givens4.hip.
//
// Output: scores[h * seq_len + pos] = S_trig + S_norm (see paper §4).
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_asym4(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    float* __restrict__ scores,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,
    float rope_theta,
    float p_q,
    int seq_len
) {
    const int pos = blockIdx.x;
    const int h = blockIdx.y;
    if (pos >= seq_len || h >= n_heads) return;
    const int tid = threadIdx.x;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int n_bands = head_dim / 2;
    const int n_rot_bands = n_rot / 2;
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 2;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const unsigned char* k_head = k_cache + (size_t)pos * k_bytes_per_pos + kv_h * k_bytes_per_head;

    const float cnorm = *(const float*)k_head;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    for (int half = 0; half < n_halves; half++) {
        const int block_tid = half * 32 + tid;
        const int base_off  = 4 + half * 64 + tid * 2;

        const unsigned int b0 = (unsigned int)k_head[base_off + 0];
        const unsigned int b1 = (unsigned int)k_head[base_off + 1];
        const int i0 = (int)((b0     ) & 0xF);
        const int i1 = (int)((b0 >> 4) & 0xF);
        const int i2 = (int)((b1     ) & 0xF);
        const int i3 = (int)((b1 >> 4) & 0xF);

        const float v0 = cnorm * TURBO_C4[i0];
        const float v1 = cnorm * TURBO_C4[i1];
        const float v2 = cnorm * TURBO_C4[i2];
        const float v3 = cnorm * TURBO_C4[i3];

        const float cA = cos_theta[block_tid * 2 + 0];
        const float sA = sin_theta[block_tid * 2 + 0];
        const float cB = cos_theta[block_tid * 2 + 1];
        const float sB = sin_theta[block_tid * 2 + 1];

        // Inverse Givens on each pair (see givens_common.h::givens_inverse).
        const float k0_re =  cA * v0 + sA * v1;
        const float k0_im = -sA * v0 + cA * v1;
        const float k1_re =  cB * v2 + sB * v3;
        const float k1_im = -sB * v2 + cB * v3;

        const float k0_mag = sqrtf(k0_re * k0_re + k0_im * k0_im);
        const float k0_phase = atan2f(k0_im, k0_re);
        const float k1_mag = sqrtf(k1_re * k1_re + k1_im * k1_im);
        const float k1_phase = atan2f(k1_im, k1_re);

        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int f = block_tid * 2 + sub;
            const float k_mag   = (sub == 0) ? k0_mag   : k1_mag;
            const float k_phase = (sub == 0) ? k0_phase : k1_phase;

            const float* c = c_head + f * 3;
            const float c_re  = c[0];
            const float c_im  = c[1];
            const float c_abs = c[2];
            const float c_mag   = sqrtf(c_re * c_re + c_im * c_im);
            const float c_phase = atan2f(c_im, c_re);

            float omega = 0.0f;
            if (f < n_rot_bands) {
                const float exponent = -2.0f * (float)f / (float)n_rot;
                omega = expf(exponent * logf(rope_theta));
            }

            const float angle = omega * p_q + c_phase - k_phase;
            s_trig += c_mag * k_mag * cosf(angle);

            float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
            if (r > 1.0f) r = 1.0f;
            s_norm += (1.0f - r) * c_abs * k_mag;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }
    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
