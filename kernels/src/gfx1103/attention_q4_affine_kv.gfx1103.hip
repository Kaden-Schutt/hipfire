// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS flash-decode attention over an affine 4-bit KV
// cache. Serves BOTH attention_q4kv and attention_hfq4_kv — they share an
// identical block layout: per (pos, kv_head) an 8 + head_dim/2 byte block of
// [f32 scale][f32 zero][head_dim/2 nibbles], dequant = scale*nibble + zero
// (even dim = low nibble, odd = high). Same ABI/math as those kernels but with
// a register-resident online softmax (no scores[seq_len] LDS). One wave32 per
// head; lane l owns dims { l, l+32, ... }. Zero __shared__. head_dim % 32 == 0.

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

__device__ __forceinline__ float hipfire_q4_affine(
    const unsigned char* blk, float sc, float zr, int d
) {
    const unsigned char pk = blk[8 + (d >> 1)];
    const unsigned int nib = (d & 1) ? (pk >> 4) : (pk & 0xF);
    return sc * (float)nib + zr;
}

__device__ __forceinline__ void hipfire_attn_q4_affine(
    const float* q, const unsigned char* k_cache, const unsigned char* v_cache,
    float* out, const int* pos_buf, int n_heads, int n_kv_heads, int head_dim,
    float scale_attn
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int seq_len = pos_buf[0] + 1;
    const int lane = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int bpb = 8 + head_dim / 2;
    const int bpp = n_kv_heads * bpb;
    const int kv_boff = kv_h * bpb;
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
        const unsigned char* kb = k_cache + (size_t)t * bpp + kv_boff;
        const float ksc = *(const float*)kb;
        const float kzr = *(const float*)(kb + 4);
        float part = 0.0f;
        int di = 0;
        for (int d = lane; d < head_dim; d += 32)
            part += q_reg[di++] * hipfire_q4_affine(kb, ksc, kzr, d);
        const float s = hipfire_wave_allsum32(part) * scale_attn;

        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        const unsigned char* vb = v_cache + (size_t)t * bpp + kv_boff;
        const float vsc = *(const float*)vb;
        const float vzr = *(const float*)(vb + 4);
        di = 0;
        for (int d = lane; d < head_dim; d += 32) {
            acc[di] = acc[di] * alpha + p * hipfire_q4_affine(vb, vsc, vzr, d);
            di++;
        }
        m = m_new;
    }

    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
    int di = 0;
    for (int d = lane; d < head_dim; d += 32) out_head[d] = acc[di++] * inv_l;
}
#endif  // HIPFIRE_GFX1103

extern "C" __launch_bounds__(32, 1)
__global__ void attention_q4kv_gfx1103(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache_q4,
    const unsigned char* __restrict__ v_cache_q4,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads, int n_kv_heads, int head_dim, int max_seq, float scale_attn
) {
#if HIPFIRE_GFX1103
    (void)max_seq;
    hipfire_attn_q4_affine(q, k_cache_q4, v_cache_q4, out, pos_buf,
                           n_heads, n_kv_heads, head_dim, scale_attn);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + (float)k_cache_q4[0] + (float)v_cache_q4[0] + scale_attn
            + (float)pos_buf[0] + (float)(n_kv_heads + head_dim + max_seq);
#endif
}

extern "C" __launch_bounds__(32, 1)
__global__ void attention_hfq4_kv_gfx1103(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads, int n_kv_heads, int head_dim, int max_seq, float scale_attn
) {
#if HIPFIRE_GFX1103
    (void)max_seq;
    hipfire_attn_q4_affine(q, k_cache, v_cache, out, pos_buf,
                           n_heads, n_kv_heads, head_dim, scale_attn);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + (float)k_cache[0] + (float)v_cache[0] + scale_attn
            + (float)pos_buf[0] + (float)(n_kv_heads + head_dim + max_seq);
#endif
}
