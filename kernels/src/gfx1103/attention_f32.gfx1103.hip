// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS flash-decode attention over an F32 KV cache.
// Same ABI/math as kernels/src/attention.hip (attention_f32), but removes the
// `scores[seq_len]` LDS buffer via a register-resident online softmax. One
// wave32 per head; lane l owns output dims { l, l+32, ... }. Zero __shared__.
// See attention_q8_0_kv.gfx1103.hip for the pattern. head_dim % 32 == 0.

#include <hip/hip_runtime.h>

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#define HIPFIRE_ATTN_MAX_DL 16  // head_dim up to 512

#if HIPFIRE_GFX1103
__device__ __forceinline__ float hipfire_wave_allsum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, 32);
    return v;
}
#endif

extern "C" __launch_bounds__(32, 1)
__global__ void attention_f32_gfx1103(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int seq_len = pos_buf[0] + 1;
    const int lane = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int kv_off = kv_h * head_dim;
    const float* q_head = q + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) {
        q_reg[ndl] = q_head[d];
        acc[ndl] = 0.0f;
        ndl++;
    }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_cache + (size_t)t * kv_stride + kv_off;
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) part += q_reg[di++] * k_t[d];
        const float s = hipfire_wave_allsum32(part) * scale;

        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        const float* v_t = v_cache + (size_t)t * kv_stride + kv_off;
        di = 0;
        for (int d = lane; d < head_dim; d += 32) {
            acc[di] = acc[di] * alpha + p * v_t[d];
            di++;
        }
        m = m_new;
    }

    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) out_head[d] = acc[di++] * inv_l;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + k_cache[0] + v_cache[0] + scale + (float)pos_buf[0]
            + (float)(n_kv_heads + head_dim + max_seq);
#endif
}
