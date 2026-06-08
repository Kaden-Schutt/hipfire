// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gemma4 L3 fused per-head attention pre-RoPE op. Collapses the 4-launch
// sequence
//
//   rmsnorm_batched(q, q_norm_w, n_heads, head_dim)   // weighted RMSNorm
//   rmsnorm_batched(k, k_norm_w, n_kv,    head_dim)   // weighted RMSNorm
//   scale_f32(q, sqrt(head_dim))                      // Gemma4 q prescale
//   rope_partial_halved_f32(q, k, ..., n_rot_pairs)   // dual RoPE (param'd)
//
// into ONE launch. V is NOT touched (its weight-less RMSNorm and the k_eq_v
// k->v capture stay outside this kernel). Math notes:
//
//   * q_norm / k_norm are WEIGHTED RMSNorm:
//       out[i] = x[i] * weight[i] * rsqrt(sum_sq/head_dim + eps)
//     (this is the gemma4 convention — NOT the unweighted L2 norm used by the
//      DeltaNet `fused_qk_l2_norm_scale` kernel; that one would drop the
//      learned q_norm/k_norm weights).
//   * Q prescale is applied as a SEPARATE multiply after the norm
//       q[i] = (q[i] * w[i] * rms) * q_scale
//     matching the unfused `rmsnorm_batched` then `scale_f32` rounding order.
//   * RoPE pairing is HuggingFace rotate_half: pair (i, i+head_dim/2). Only the
//     first `n_rot_pairs` pairs rotate; pairs >= n_rot_pairs are NoPE
//     (pass-through). freq uses the FULL head_dim (proportional-RoPE spec).
//     Sliding layers pass n_rot_pairs = head_dim/2 (full rotate-half);
//     full layers pass n_rot_pairs = partial_rotary_factor*head_dim/2.
//
// Grid: [n_heads, 1, 1], block = 32 (one wave32). Block h normalizes+scales+
// RoPEs Q head h; if h < n_kv it ALSO normalizes+RoPEs K head h. Requires
// n_kv <= n_heads (always true for GQA). This is decode-only (single position);
// q/k laid out [n_heads*head_dim] / [n_kv*head_dim] row-major.
//
// Not byte-identical to the unfused chain (fused rsqrt vs the separate
// rmsnorm/scale/rope launches round in a different order) -- coherence-validated.
extern "C" __global__ void fused_gemma4_qk_norm_rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ q_norm_w,   // [head_dim]
    const float* __restrict__ k_norm_w,   // [head_dim]
    const int* __restrict__ pos_buf,      // device single i32 — graph-safe
    int n_heads,
    int n_kv,
    int head_dim,
    int n_rot_pairs,
    float q_scale,
    float freq_base,
    float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;   // wave32: 0..31
    const int half = head_dim / 2;
    const int pos = *pos_buf;

    // ── Q head h: weighted RMSNorm, then q_scale ──
    float* q_head = q + (long long)h * head_dim;
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_rms = rsqrtf(q_sq / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] = q_head[i] * q_norm_w[i] * q_rms;
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_scale;

    // ── K head h (only first n_kv blocks): weighted RMSNorm ──
    float* k_head = nullptr;
    if (h < n_kv) {
        k_head = k + (long long)h * head_dim;
        float k_sq = 0.0f;
        for (int i = tid; i < head_dim; i += 32)
            k_sq += k_head[i] * k_head[i];
        for (int o = 16; o > 0; o >>= 1)
            k_sq += __shfl_xor(k_sq, o);
        float k_rms = rsqrtf(k_sq / (float)head_dim + eps);
        for (int i = tid; i < head_dim; i += 32)
            k_head[i] = k_head[i] * k_norm_w[i] * k_rms;
    }

    // ── RoPE (rotate_half, first n_rot_pairs pairs) on Q (and K if present) ──
    // All norm writes above are within this same block/wave; the in-place
    // pair reads below need the post-norm values, so sync the wave first.
    __syncthreads();
    for (int pair = tid; pair < n_rot_pairs; pair += 32) {
        const float freq =
            1.0f / powf(freq_base, (float)(2 * pair) / (float)head_dim);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);

        const float q0 = q_head[pair];
        const float q1 = q_head[pair + half];
        q_head[pair]        = q0 * cos_a - q1 * sin_a;
        q_head[pair + half] = q0 * sin_a + q1 * cos_a;

        if (k_head != nullptr) {
            const float k0 = k_head[pair];
            const float k1 = k_head[pair + half];
            k_head[pair]        = k0 * cos_a - k1 * sin_a;
            k_head[pair + half] = k0 * sin_a + k1 * cos_a;
        }
    }
}
