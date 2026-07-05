// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS batched Q8_0 causal/tree attention.
// Same ABI/math as kernels/src/attention_q8_0_kv_batched.hip: grid
// [n_heads, batch_size], one block per (head, batch row). Causal window is
// positions[b]+1, or tree mode (tree_bias != null) over block_start+block_cols
// with per-key bias for keys in [block_start, block_start+block_cols). The
// generic kernel stages scores[seq_len] in LDS (unbounded → wedge risk). This
// variant runs a register-resident online softmax: one wave32 per (head, row),
// lane l owns dims { l, l+32, ... }, zero __shared__. head_dim % 32 == 0.

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
__global__ void attention_q8_0_kv_batched_gfx1103(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int block_start,
    int block_cols
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[b] + 1);

    const int kv_h = h / (n_heads / n_kv_heads);
    const int bph = head_dim / 32;
    const int total_bpp = n_kv_heads * bph;
    const int kv_block0 = kv_h * bph;
    const int q_stride = n_heads * head_dim;
    const float* q_head = q + (size_t)b * q_stride + h * head_dim;

    float q_reg[HIPFIRE_ATTN_MAX_DL];
    float acc[HIPFIRE_ATTN_MAX_DL];
    for (int bi = 0; bi < bph; bi++) {
        q_reg[bi] = q_head[bi * 32 + lane];
        acc[bi] = 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const size_t base = (size_t)t * total_bpp * 34;
        float part = 0.0f;
        for (int bi = 0; bi < bph; bi++) {
            const unsigned char* kb = k_cache + base + (size_t)(kv_block0 + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            part += q_reg[bi] * ks * (float)((signed char)kb[2 + lane]);
        }
        float s = hipfire_wave_allsum32(part) * scale_attn;
        if (tree_mode && t >= block_start) {
            s += tree_bias[b * block_cols + (t - block_start)];
        }

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
        out[0] = q[0] + (float)k_cache[0] + (float)v_cache[0] + (float)positions[0]
            + scale_attn + (float)(n_kv_heads + head_dim + max_seq + block_start + block_cols)
            + (tree_bias ? tree_bias[0] : 0.0f);
#endif
}
