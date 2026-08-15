// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring for adaptive FWHT{2,3,4} post-RoPE K caches.
// The packed rows share storage with asym{2,3,4}, but require an inverse
// signed FWHT before their complex bands can be used by TriAttention.
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1]. head_dim must be 256.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

__device__ __forceinline__ void triattn_fwht_band(
    float re,
    float im,
    const float* __restrict__ center,
    int band,
    int n_rot_bands,
    int n_rot,
    float rope_theta,
    float p_q,
    float& s_trig,
    float& s_norm
) {
    const float k_mag = sqrtf(re * re + im * im);
    const float k_phase = atan2f(im, re);
    const float c_re = center[0];
    const float c_im = center[1];
    const float c_abs = center[2];
    const float c_mag = sqrtf(c_re * c_re + c_im * c_im);
    const float c_phase = atan2f(c_im, c_re);

    float omega = 0.0f;
    if (band < n_rot_bands) {
        const float exponent = -2.0f * (float)band / (float)n_rot;
        omega = expf(exponent * logf(rope_theta));
    }
    s_trig += c_mag * k_mag * cosf(omega * p_q + c_phase - k_phase);
    float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
    if (r > 1.0f) r = 1.0f;
    s_norm += (1.0f - r) * c_abs * k_mag;
}

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_fwht(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ scores,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,
    float rope_theta,
    float p_q,
    int seq_len,
    int bits
) {
    const int pos = blockIdx.x;
    const int h = blockIdx.y;
    if (pos >= seq_len || h >= n_heads || head_dim != 256) return;
    const int tid = threadIdx.x;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int n_bands = head_dim / 2;
    const int n_rot_bands = n_rot / 2;
    const int bytes_per_head = 4 + (head_dim * bits) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const unsigned char* row =
        k_cache + (size_t)pos * bytes_per_pos + kv_h * bytes_per_head;
    const float cnorm = *(const float*)row;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;
    if (bits == 3) {
        const unsigned char* packed_base = row + 4 + tid * 3;
        const unsigned int packed = (unsigned int)packed_base[0]
            | ((unsigned int)packed_base[1] << 8)
            | ((unsigned int)packed_base[2] << 16);
        float v0 = cnorm * TURBO_C3_256[(packed      ) & 7];
        float v1 = cnorm * TURBO_C3_256[(packed >>  3) & 7];
        float v2 = cnorm * TURBO_C3_256[(packed >>  6) & 7];
        float v3 = cnorm * TURBO_C3_256[(packed >>  9) & 7];
        float v4 = cnorm * TURBO_C3_256[(packed >> 12) & 7];
        float v5 = cnorm * TURBO_C3_256[(packed >> 15) & 7];
        float v6 = cnorm * TURBO_C3_256[(packed >> 18) & 7];
        float v7 = cnorm * TURBO_C3_256[(packed >> 21) & 7];
        fwht_shfl_inverse_256(
            v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);
        const float vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int band = tid * 4 + pair;
            triattn_fwht_band(
                vals[pair * 2], vals[pair * 2 + 1], c_head + band * 3,
                band, n_rot_bands, n_rot, rope_theta, p_q, s_trig, s_norm);
        }
    } else {
        // FWHT2/FWHT4 use two independent 128-wide transforms. Each lane owns
        // four consecutive values (two complex bands) in each half.
        const int bytes_per_lane = bits / 2;
        const int bytes_per_half = 128 * bits / 8;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned char* packed_base =
                row + 4 + half * bytes_per_half + tid * bytes_per_lane;
            unsigned int packed = (unsigned int)packed_base[0];
            if (bits == 4) packed |= ((unsigned int)packed_base[1] << 8);
            float v0, v1, v2, v3;
            if (bits == 2) {
                v0 = cnorm * TURBO_C2[(packed      ) & 3];
                v1 = cnorm * TURBO_C2[(packed >>  2) & 3];
                v2 = cnorm * TURBO_C2[(packed >>  4) & 3];
                v3 = cnorm * TURBO_C2[(packed >>  6) & 3];
            } else {
                v0 = cnorm * TURBO_C4[(packed      ) & 15];
                v1 = cnorm * TURBO_C4[(packed >>  4) & 15];
                v2 = cnorm * TURBO_C4[(packed >>  8) & 15];
                v3 = cnorm * TURBO_C4[(packed >> 12) & 15];
            }
            fwht_shfl_inverse(v0, v1, v2, v3, signs1, signs2, tid);
            const int band0 = half * 64 + tid * 2;
            triattn_fwht_band(
                v0, v1, c_head + band0 * 3, band0, n_rot_bands, n_rot,
                rope_theta, p_q, s_trig, s_norm);
            triattn_fwht_band(
                v2, v3, c_head + (band0 + 1) * 3, band0 + 1,
                n_rot_bands, n_rot, rope_theta, p_q, s_trig, s_norm);
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }
    if (tid == 0) scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
}
