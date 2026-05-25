// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on a Q8 post-RoPE K cache (arXiv:2604.04921).
//
// For each cached key at position p_k, compute S_total(k, Δ) = S_trig + S_norm,
// where:
//   S_trig = Σ_f ||E[q_f]|| · ||k_f|| · cos(ω_f · p_q + arg(E[q_f]) - arg(k_f))
//   S_norm = Σ_f (1 - R_f) · E[||q_f||] · ||k_f||
//
// The scoring uses our post-RoPE simplification: since we store post-RoPE
// K and `k_pre_f = k_post_f · e^{-iω_f·p_k}`, the position-dependent phase
// cancels out of S_trig, leaving a formula that depends only on p_q
// (the current query position) and the stored k_post_f.
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1] (one WAVE32 per score).
// Output: scores[h * seq_len + p_k].
//
// Q8 K layout (matches attention_flash_q8_0_tile.hip):
//   K[pos, head, :] = blocks of 32 int8 + 1 f16 scale (34 bytes per block).
//   row_stride = n_kv_heads × blocks_per_head × 34.
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_q8(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,   // [n_heads × n_bands × 3] (eq_re, eq_im, e_abs_q)
    float* __restrict__ scores,          // [n_heads × seq_len]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,                           // partial_rotary_factor × head_dim
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
    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    const int row_stride = total_blocks * 34;

    const unsigned char* k_pos = k_cache + (size_t)pos * row_stride;
    const unsigned char* k_head = k_pos + kv_h * blocks_per_head * 34;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    // Each thread handles n_bands/32 bands. For head_dim=256 → 4 bands/thread.
    for (int f = tid; f < n_bands; f += 32) {
        // Complex pair (2f, 2f+1) lies within a single 32-wide block.
        const int dim0 = 2 * f;
        const int block_idx = dim0 / 32;
        const int within = dim0 % 32;
        const unsigned char* blk = k_head + block_idx * 34;
        const float scale = (float)(*(const _Float16*)blk);
        const int q0 = (signed char)blk[2 + within];
        const int q1 = (signed char)blk[2 + within + 1];
        const float k_re = scale * (float)q0;
        const float k_im = scale * (float)q1;
        const float k_mag = sqrtf(k_re * k_re + k_im * k_im);
        const float k_phase = atan2f(k_im, k_re);

        const float* c = c_head + f * 3;
        const float c_re = c[0];
        const float c_im = c[1];
        const float c_abs = c[2];
        const float c_mag = sqrtf(c_re * c_re + c_im * c_im);
        const float c_phase = atan2f(c_im, c_re);

        float omega = 0.0f;
        if (f < n_rot_bands) {
            // ω_f = θ^(-2f / n_rot). Bands beyond n_rot/2 are unrotated.
            const float exponent = -2.0f * (float)f / (float)n_rot;
            omega = expf(exponent * logf(rope_theta));
        }

        const float angle = omega * p_q + c_phase - k_phase;
        s_trig += c_mag * k_mag * cosf(angle);

        // R_f = ||E[q_f]|| / E[||q_f||]. Clamped to [0, 1].
        float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
        if (r > 1.0f) r = 1.0f;
        s_norm += (1.0f - r) * c_abs * k_mag;
    }

    // Warp-reduce to thread 0.
    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }

    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
