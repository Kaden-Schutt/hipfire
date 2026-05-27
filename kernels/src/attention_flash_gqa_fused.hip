// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Single-launch GQA flash decode: one block per kv_head streams the whole KV
// cache once, accumulates online-softmax (m, l, O) for all `kv_group` query
// heads in LDS, normalizes, and writes O directly. Removes the partials DRAM
// round-trip and the second (reduce) launch vs attention_flash_gqa. Trade: grid
// = n_kv_heads only (2 for dots.ocr 12:2), so this wins on launch/traffic but
// loses occupancy — a probe of which is the decode floor. F32 cache.
extern "C" __global__ void attention_flash_gqa_fused(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,            // [n_heads, head_dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float scale
) {
    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;
    const int kv_group = n_heads / n_kv_heads;   // 6
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kd = n_kv_heads * head_dim;        // cache row stride

    // LDS: per-group running output O[g][d], plus a scratch row for reductions.
    extern __shared__ float sdata[];
    float* O = sdata;                  // [kv_group * head_dim]
    float* ws = O + kv_group * head_dim;  // [nthreads]

    for (int g = 0; g < kv_group; g++)
        for (int d = tid; d < head_dim; d += nthreads) O[g * head_dim + d] = 0.0f;
    __syncthreads();

    for (int g = 0; g < kv_group; g++) {
        const int h = kv_h * kv_group + g;
        const float* q_head = q + h * head_dim;
        float* Og = O + g * head_dim;
        float m = -1e30f, l = 0.0f;
        // online over positions; each thread carries running (m,l), O shared
        for (int pos = 0; pos < seq_len; pos++) {
            const float* k_t = k_cache + pos * kd + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads) dot += q_head[d] * k_t[d];
            ws[tid] = dot; __syncthreads();
            for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] += ws[tid + s]; __syncthreads(); }
            float s = ws[0] * scale; __syncthreads();
            float m_new = fmaxf(m, s);
            float corr = expf(m - m_new);
            float p = expf(s - m_new);
            l = l * corr + p;
            const float* v_t = v_cache + pos * kd + kv_h * head_dim;
            for (int d = tid; d < head_dim; d += nthreads) Og[d] = Og[d] * corr + p * v_t[d];
            m = m_new;
        }
        float inv = 1.0f / l;
        float* out_h = out + h * head_dim;
        for (int d = tid; d < head_dim; d += nthreads) out_h[d] = Og[d] * inv;
        __syncthreads();
    }
}
