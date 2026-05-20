// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on an asym3 post-RoPE K cache (arXiv:2604.04921).
//
// asym3 stores K as: [4B cnorm][32 threads × 3 bytes packed 3-bit indices].
// To recover post-RoPE K per band for scoring, each thread:
//   1. Reads its 3 bytes (8 × 3-bit indices) and dequantizes via the
//      Lloyd-Max N(0,1/256) codebook (same table asym3 write used).
//   2. Multiplies by cnorm (the shared magnitude-correction scalar stored
//      at the head block head).
//   3. Inverse-applies the per-band Givens rotation that the writer used
//      to spread mass across dims — this un-mixing recovers post-RoPE
//      (complex) K at each band, so arg/norm can be read directly.
//
// Layout (matches kv_cache_write_asym_k_givens3.hip):
//   bytes_per_head = 4 + (head_dim * 3) / 8     // 100 for hd=256
//   per-thread owns 8 consecutive dims = 4 complex bands (tid*4 .. tid*4+3)
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_asym3(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,       // [n_heads × n_bands × 3]
    const float* __restrict__ cos_theta,     // [n_bands] Givens angles (shared across layers)
    const float* __restrict__ sin_theta,     // [n_bands]
    float* __restrict__ scores,              // [n_heads × seq_len]
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

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const unsigned char* k_head = k_cache + (size_t)pos * k_bytes_per_pos + kv_h * k_bytes_per_head;

    // Shared per-head magnitude-restore factor.
    const float cnorm = *(const float*)k_head;

    // This thread's 3 packed bytes → 8 quantized scalar values.
    const unsigned char* base = k_head + 4 + tid * 3;
    const unsigned int packed = (unsigned int)base[0]
                              | ((unsigned int)base[1] << 8)
                              | ((unsigned int)base[2] << 16);
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int idx = (packed >> (i * 3)) & 7;
        v[i] = cnorm * TURBO_C3_256[idx];
    }

    // Inverse Givens per band pair. Forward was:
    //   a' = a*cos(θ) - b*sin(θ);  b' = a*sin(θ) + b*cos(θ)
    // So inverse (rotate by -θ):
    //   a = a'*cos(θ) + b'*sin(θ);  b = -a'*sin(θ) + b'*cos(θ)
    const int b0 = tid * 4;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int f = b0 + j;
        const float cb = cos_theta[f];
        const float sb = sin_theta[f];

        const float a_gv = v[j * 2 + 0];
        const float b_gv = v[j * 2 + 1];
        const float k_re =  cb * a_gv + sb * b_gv;
        const float k_im = -sb * a_gv + cb * b_gv;
        const float k_mag = sqrtf(k_re * k_re + k_im * k_im);
        const float k_phase = atan2f(k_im, k_re);

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

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }

    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
