// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS batched causal attention (F32 KV, prefill).
// Same ABI/math as kernels/src/attention_causal_batched.hip: grid is
// [n_heads, seq_len]; block (h, qpos) computes the causal attention for query
// position qpos over K/V positions 0..qpos. The generic kernel stages
// scores[qpos+1] in LDS (scales with position → the same unbounded-LDS wedge
// risk as the decode kernels). This variant runs a register-resident online
// softmax: one wave32 per (head, query), lane l owns dims { l, l+32, ... },
// zero __shared__. head_dim % 32 == 0.

#include <hip/hip_runtime.h>

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#define HIPFIRE_ATTN_MAX_DL 16

#if HIPFIRE_GFX1103
__device__ __forceinline__ float hipfire_wave_allsum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, 32);
    return v;
}
#endif

extern "C" __launch_bounds__(32, 1)
__global__ void attention_causal_batched_gfx1103(
    const float* __restrict__ Q,    // [seq_len x n_heads x head_dim]
    const float* __restrict__ K,    // [seq_len x n_kv_heads x head_dim]
    const float* __restrict__ V,    // [seq_len x n_kv_heads x head_dim]
    float* __restrict__ out,        // [seq_len x n_heads x head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    const int qpos = blockIdx.y;
    if (h >= n_heads || qpos >= seq_len) return;
    const int lane = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;
    const int kv_off = kv_h * head_dim;
    const int ctx_len = qpos + 1;  // causal: attend to 0..qpos
    const float* q_head = Q + (size_t)qpos * q_stride + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) { q_reg[ndl] = q_head[d]; acc[ndl] = 0.0f; ndl++; }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < ctx_len; t++) {
        const float* k_t = K + (size_t)t * kv_stride + kv_off;
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) part += q_reg[di++] * k_t[d];
        const float s = hipfire_wave_allsum32(part) * scale;

        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        const float* v_t = V + (size_t)t * kv_stride + kv_off;
        di = 0;
        for (int d = lane; d < head_dim; d += 32) { acc[di] = acc[di] * alpha + p * v_t[d]; di++; }
        m = m_new;
    }

    float* out_head = out + (size_t)qpos * q_stride + h * head_dim;
    const float inv_l = 1.0f / l;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) out_head[d] = acc[di++] * inv_l;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = Q[0] + K[0] + V[0] + scale
            + (float)(seq_len + n_kv_heads + head_dim);
#endif
}
