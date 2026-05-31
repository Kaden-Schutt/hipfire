// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Warp-cooperative GQA decode attention: 32 threads of a warp share one
// query head and cooperatively compute dot-products + V-accumulates.
//
// Key design: instead of the current pattern where each thread handles one
// token (stride-256 uncoalesced K-access), all 32 threads of a warp load
// 4 consecutive floats from the SAME K-row → perfectly coalesced 128-byte
// loads, then warp-shuffle-reduce to get the full dot product.
//
// Grid:  [n_kv_heads, n_chunks]
// Block: (kv_group × 32) threads = one warp per query head
//
// Each warp handles one head. All warps in a block share the same chunk
// of K/V (L2-cached after first load). Online softmax in registers; no
// score-array LDS needed. Writes partial (m, l, O[d]) for the existing
// attention_flash_reduce to combine chunks.
//
// head_dim must be 128 (4 floats per lane × 32 lanes = 128).
extern "C" __global__ void attention_gqa_warp(
    const float* __restrict__ q,           // [n_heads, head_dim]
    const float* __restrict__ k_cache,     // [max_seq, n_kv_heads * head_dim]
    const float* __restrict__ v_cache,     // [max_seq, n_kv_heads * head_dim]
    float* __restrict__ partials,          // [n_heads, n_chunks, 2 + head_dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float scale, int chunk_size
) {
    const int kv_h   = blockIdx.x;
    const int chunk_id = blockIdx.y;
    if (kv_h >= n_kv_heads) return;
    const int kv_group = n_heads / n_kv_heads;   // 6 for dots.ocr
    const int warp_id  = threadIdx.x >> 5;        // which head in the group
    const int lane     = threadIdx.x & 31;

    if (warp_id >= kv_group) return;
    const int h = kv_h * kv_group + warp_id;

    const int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;
    if (chunk_start >= seq_len) return;

    const int kd = n_kv_heads * head_dim;          // KV cache row stride
    const float* q_head = q + h * head_dim;

    // Load query into registers: 4 floats per lane, perfectly coalesced
    // within the warp (q_head is contiguous 512 bytes).
    float q0 = q_head[lane * 4 + 0];
    float q1 = q_head[lane * 4 + 1];
    float q2 = q_head[lane * 4 + 2];
    float q3 = q_head[lane * 4 + 3];

    // Online softmax state — stays in registers
    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const float* k_base = k_cache + kv_h * head_dim;
    const float* v_base = v_cache + kv_h * head_dim;

    for (int t = chunk_start; t < chunk_end; t++) {
        // Coalesced K-load: 32 threads × 4 floats from the SAME K-row
        const float* k_row = k_base + (long)t * kd;
        float k0 = k_row[lane * 4 + 0];
        float k1 = k_row[lane * 4 + 1];
        float k2 = k_row[lane * 4 + 2];
        float k3 = k_row[lane * 4 + 3];

        // Warp-reduce dot product: 5 shuffle-adds of 32 lanes
        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_xor(dot, offset);
        // dot now holds the full q·k in all lanes

        float s = dot * scale;

        // Online softmax update
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p     = __expf(s - m_new);
        l = l * alpha + p;

        // Scale existing accumulator
        o0 *= alpha;
        o1 *= alpha;
        o2 *= alpha;
        o3 *= alpha;

        // Coalesced V-load + accumulate
        const float* v_row = v_base + (long)t * kd;
        float v0 = v_row[lane * 4 + 0];
        float v1 = v_row[lane * 4 + 1];
        float v2 = v_row[lane * 4 + 2];
        float v3 = v_row[lane * 4 + 3];

        o0 += p * v0;
        o1 += p * v1;
        o2 += p * v2;
        o3 += p * v3;

        m = m_new;
    }

    // Write partials for the reduce kernel: [h * n_chunks + chunk_id, 2 + head_dim]
    const int n_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int stride = 2 + head_dim;
    float* partial = partials + (h * n_chunks + chunk_id) * stride;

    partial[0] = m;
    partial[1] = l;
    partial[2 + lane * 4 + 0] = o0;
    partial[2 + lane * 4 + 1] = o1;
    partial[2 + lane * 4 + 2] = o2;
    partial[2 + lane * 4 + 3] = o3;
}
