// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched decode-with-history attention over an F32 KV cache.
//
// Drives block-parallel spec-decode verify on qwen2 (arch_id=7): a block of
// `batch` query rows, each at absolute KV position `base + i`, attends to the
// FULL prior history [0 .. base+i] read directly from the persistent F32 KV
// cache (k_cache/v_cache, indexed by ABSOLUTE position). The caller must have
// already written the block's own post-RoPE K/V into the cache at positions
// [base .. base+batch) before launching, so each query sees its own causal
// prefix within the block plus all prompt/accepted history.
//
// This is exactly the per-token `attention_flash` math that `forward_step`
// runs sequentially, but one block per (head, query-row) instead of one
// sequential launch per token — so the result is BYTE-IDENTICAL to the
// sequential verify (same scores, same single-pass softmax, same V-weighted
// sum), only block-parallel.
//
// Grid:  [n_heads, batch, 1]   one block per (query head, block row)
// Block: [nthreads, 1, 1]
// Shmem: (max_ctx_len + nthreads) floats  (scores[ctx_len] + workspace[nthreads])
extern "C" __global__ void attention_decode_batched_history(
    const float* __restrict__ Q,        // [batch × n_heads × head_dim]  block queries
    const float* __restrict__ k_cache,  // [max_seq × n_kv_heads × head_dim]  ABSOLUTE pos
    const float* __restrict__ v_cache,  // [max_seq × n_kv_heads × head_dim]  ABSOLUTE pos
    float* __restrict__ out,            // [batch × n_heads × head_dim]
    int batch,
    int base,                            // absolute position of block row 0
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale
) {
    const int h = blockIdx.x;        // query head
    const int row = blockIdx.y;      // block row 0..batch-1
    if (h >= n_heads || row >= batch) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int q_stride = n_heads * head_dim;       // per-row stride in Q/out
    const int kv_stride = n_kv_heads * head_dim;   // per-position stride in cache

    // Query for this (row, head). Its absolute position is base + row, so it
    // attends to absolute positions [0 .. base+row] (causal).
    const float* q_head = Q + row * q_stride + h * head_dim;
    const int ctx_len = base + row + 1;

    extern __shared__ float sdata[];
    float* scores = sdata;             // [ctx_len]
    float* ws = sdata + ctx_len;       // [nthreads] reduction workspace

    // Phase 1: QK^T over the full causal history (absolute positions).
    float lmax = -1e30f;
    for (int t = tid; t < ctx_len; t += nthreads) {
        const float* k_t = k_cache + (size_t)t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_head[d] * k_t[d];
        float s = dot * scale;
        scores[t] = s;
        lmax = fmaxf(lmax, s);
    }

    ws[tid] = lmax;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]); __syncthreads(); }
    float max_val = ws[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int t = tid; t < ctx_len; t += nthreads) { float e = expf(scores[t] - max_val); scores[t] = e; lsum += e; }
    ws[tid] = lsum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] += ws[tid + s]; __syncthreads(); }
    float sv = ws[0];
    __syncthreads();
    for (int t = tid; t < ctx_len; t += nthreads) scores[t] /= sv;
    __syncthreads();

    // Phase 2: weighted sum of V over the full history.
    float* out_head = out + row * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            const float* v_t = v_cache + (size_t)t * kv_stride + kv_h * head_dim;
            val += scores[t] * v_t[d];
        }
        out_head[d] = val;
    }
}
