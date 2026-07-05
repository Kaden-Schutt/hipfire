// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS flash-decode attention over an int8c KV cache.
// Same ABI/math as kernels/src/attention_int8c_kv.hip: per (pos, kv_head) an
// 8 + head_dim byte block of [f32 scale][4 pad][head_dim int8], dequant =
// scale * (int8). Register-resident online softmax, no scores[seq_len] LDS.
// One wave32 per head; lane l owns dims { l, l+32, ... }. head_dim % 32 == 0.

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
__global__ void attention_int8c_kv_gfx1103(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
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
    const int bph = 8 + head_dim;
    const int bpp = n_kv_heads * bph;
    const int kv_boff = kv_h * bph;
    const float* q_head = q + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    int ndl = 0;
    for (int d = lane; d < head_dim; d += 32) { q_reg[ndl] = q_head[d]; acc[ndl] = 0.0f; ndl++; }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + (size_t)t * bpp + kv_boff;
        const float ksc = *(const float*)kb;
        const signed char* kd = (const signed char*)(kb + 8);
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32) part += q_reg[di++] * (float)kd[d];
        const float s = hipfire_wave_allsum32(part) * ksc * scale_attn;

        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        const unsigned char* vb = v_cache + (size_t)t * bpp + kv_boff;
        const float vsc = *(const float*)vb;
        const signed char* vd = (const signed char*)(vb + 8);
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
        out[0] = q[0] + (float)k_cache[0] + (float)v_cache[0] + scale_attn
            + (float)pos_buf[0] + (float)(n_kv_heads + head_dim + max_seq);
#endif
}
