// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS routed batched Q8_0 causal attention.
// Same ABI/math as kernels/src/attention_q8_0_routed_batched.hip: grid
// [n_heads, batch_size]; each row picks its session K/V cache from pointer
// tables, then attends causally over positions[b]+1. Register-resident online
// softmax replaces the scores[seq_len] LDS. One wave32 per (head, row), zero
// __shared__. head_dim % 32 == 0.

#include <hip/hip_runtime.h>
#include <stdint.h>

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
__global__ void attention_q8_0_routed_batched_gfx1103(
    const float* __restrict__ q,
    const uint64_t* __restrict__ k_ptrs,
    const uint64_t* __restrict__ v_ptrs,
    float* __restrict__ out,
    const int* __restrict__ row_session_indices,
    const int* __restrict__ positions,
    int ptr_layer_stride,
    int layer_index,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int session_idx = row_session_indices[b];
    const unsigned char* k_cache = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(k_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index]));
    const unsigned char* v_cache = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(v_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index]));

    const int seq_len = positions[b] + 1;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int bph = head_dim / 32;
    const int total_bpp = n_kv_heads * bph;
    const int kv_block0 = kv_h * bph;
    const int q_stride = n_heads * head_dim;
    const float* q_head = q + (size_t)b * q_stride + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    for (int bi = 0; bi < bph; bi++) { q_reg[bi] = q_head[bi * 32 + lane]; acc[bi] = 0.0f; }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const size_t base = (size_t)t * total_bpp * 34;
        float part = 0.0f;
        for (int bi = 0; bi < bph; bi++) {
            const unsigned char* kb = k_cache + base + (size_t)(kv_block0 + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            part += q_reg[bi] * ks * (float)((signed char)kb[2 + lane]);
        }
        const float s = hipfire_wave_allsum32(part) * scale_attn;
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;
        for (int bi = 0; bi < bph; bi++) {
            const unsigned char* vb = v_cache + base + (size_t)(kv_block0 + bi) * 34;
            const float vs = (float)*((const _Float16*)vb);
            acc[bi] = acc[bi] * alpha + p * vs * (float)((signed char)vb[2 + lane]);
        }
        m = m_new;
    }

    float* out_head = out + (size_t)b * q_stride + h * head_dim;
    const float inv_l = 1.0f / l;
    for (int bi = 0; bi < bph; bi++) out_head[bi * 32 + lane] = acc[bi] * inv_l;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0)
        out[0] = q[0] + (float)k_ptrs[0] + (float)v_ptrs[0] + (float)row_session_indices[0]
            + (float)positions[0] + (float)(ptr_layer_stride + layer_index + n_kv_heads
            + head_dim + max_seq) + scale_attn;
#endif
}
