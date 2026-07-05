// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS flash-decode attention over an int8 KV cache with
// split scale arrays. Same ABI/math as kernels/src/attention_int8_kv.hip:
// k_vals/v_vals are [max_seq x kv_dim] int8, k_scales/v_scales are
// [max_seq x n_kv_heads] f32; dequant = scale * (int8). Register-resident
// online softmax, no scores[seq_len] LDS. One wave32 per head; lane l owns
// dims { l, l+32, ... }. head_dim % 32 == 0.

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
__global__ void attention_int8_kv_gfx1103(
    const float* __restrict__ q,
    const signed char* __restrict__ k_vals,
    const float* __restrict__ k_scales,
    const signed char* __restrict__ v_vals,
    const float* __restrict__ v_scales,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads, int n_kv_heads, int head_dim, int max_seq, float scale_attn
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int seq_len = pos_buf[0] + 1;
    const int lane = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int kv_dim = n_kv_heads * head_dim;
    const int kv_off = kv_h * head_dim;
    const float* q_head = q + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) { q_reg[ndl] = q_head[d]; acc[ndl] = 0.0f; ndl++; }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float ksc = k_scales[t * n_kv_heads + kv_h];
        const signed char* kd = k_vals + (size_t)t * kv_dim + kv_off;
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) part += q_reg[di++] * (float)kd[d];
        const float s = hipfire_wave_allsum32(part) * ksc * scale_attn;

        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        const float vsc = v_scales[t * n_kv_heads + kv_h];
        const signed char* vd = v_vals + (size_t)t * kv_dim + kv_off;
        const float pv = p * vsc;
        di = 0;
        for (int d = lane; d < head_dim; d += 32) { acc[di] = acc[di] * alpha + pv * (float)vd[d]; di++; }
        m = m_new;
    }

    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) out_head[d] = acc[di++] * inv_l;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + (float)k_vals[0] + k_scales[0] + (float)v_vals[0] + v_scales[0]
            + scale_attn + (float)pos_buf[0] + (float)(n_kv_heads + head_dim + max_seq);
#endif
}
