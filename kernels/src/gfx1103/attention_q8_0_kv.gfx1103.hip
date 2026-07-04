// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1103 (Phoenix) no-LDS flash-decode attention over a Q8_0 KV cache.
//
// Same ABI and math contract as kernels/src/attention_q8_0_kv.hip, but removes
// that kernel's `scores[seq_len]` LDS buffer — which scales with context length
// and, under graph capture (sized to max_seq), can exceed the 64 KB LDS cap and
// wedge the Phoenix GPU (see AGENTS.local.md LDS hazard).
//
// Strategy: one wave32 per head, ZERO __shared__. Each lane owns element `lane`
// of every 32-wide Q8_0 block, i.e. output dims { lane, lane+32, ... } — one
// accumulator register per block (head_dim/32 of them). For each KV position a
// lane computes its partial Q·K dot, an __shfl_xor butterfly all-reduces it to
// the full score on every lane, and a single-pass ONLINE softmax updates the
// register state (running max m, running denom l, and the output accumulators)
// without ever materialising the score vector.
//
// Requires head_dim % 32 == 0 (dispatch guards this; Qwen head_dim=128). The
// Q8_0 block layout matches the generic kernel: 34 bytes per 32 weights =
// [f16 scale][32 × int8], `n_kv_heads * head_dim/32` blocks per position.

#include <hip/hip_runtime.h>

#if defined(__gfx1103__)
#define HIPFIRE_GFX1103 1
#else
#define HIPFIRE_GFX1103 0
#endif

#if HIPFIRE_GFX1103
// Butterfly all-reduce: every lane receives the 32-lane sum.
__device__ __forceinline__ float hipfire_wave_allsum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, 32);
    }
    return v;
}
#endif

#define HIPFIRE_Q8_KV_MAX_BPH 16  // head_dim up to 512

extern "C" __launch_bounds__(32, 1)
__global__ void attention_q8_0_kv_gfx1103(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
#if HIPFIRE_GFX1103
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int seq_len = pos_buf[0] + 1;
    const int lane = threadIdx.x;            // 0..31 (single wave32 block)
    const int bph = head_dim / 32;           // blocks per head (dims/lane)
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int total_blocks_per_pos = n_kv_heads * bph;
    const int kv_head_block_start = kv_h * bph;

    const float* q_head = q + h * head_dim;
    // Each lane preloads its element of every block: q[bi*32 + lane].
    float q_reg[HIPFIRE_Q8_KV_MAX_BPH];
    float acc[HIPFIRE_Q8_KV_MAX_BPH];
#pragma unroll 1
    for (int bi = 0; bi < bph; bi++) {
        q_reg[bi] = q_head[bi * 32 + lane];
        acc[bi] = 0.0f;
    }

    float m = -1e30f;   // running max
    float l = 0.0f;     // running softmax denominator

    for (int t = 0; t < seq_len; t++) {
        const size_t pos_base = (size_t)t * total_blocks_per_pos * 34;
        // Partial Q·K dot over this lane's elements, then all-reduce.
        float part = 0.0f;
#pragma unroll 1
        for (int bi = 0; bi < bph; bi++) {
            const unsigned char* kb =
                k_cache + pos_base + (size_t)(kv_head_block_start + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            const float kq = (float)((signed char)kb[2 + lane]);
            part += q_reg[bi] * ks * kq;
        }
        const float s = hipfire_wave_allsum32(part) * scale_attn;

        // Online-softmax update (identical scalars on every lane).
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + p;

        // Accumulate this lane's V elements with the running rescale.
#pragma unroll 1
        for (int bi = 0; bi < bph; bi++) {
            const unsigned char* vb =
                v_cache + pos_base + (size_t)(kv_head_block_start + bi) * 34;
            const float vs = (float)*((const _Float16*)vb);
            const float vq = (float)((signed char)vb[2 + lane]);
            acc[bi] = acc[bi] * alpha + p * vs * vq;
        }
        m = m_new;
    }

    float* out_head = out + h * head_dim;
    const float inv_l = 1.0f / l;
#pragma unroll 1
    for (int bi = 0; bi < bph; bi++) {
        out_head[bi * 32 + lane] = acc[bi] * inv_l;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_heads == 0) {
        out[0] = q[0] + (float)k_cache[0] + (float)v_cache[0] + scale_attn
            + (float)pos_buf[0] + (float)(n_kv_heads + head_dim + max_seq);
    }
#endif
}
