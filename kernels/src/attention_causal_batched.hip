// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_causal_batched(
    const float* __restrict__ Q,    // [seq_len × n_heads × head_dim]
    const float* __restrict__ K,    // [seq_len × n_kv_heads × head_dim]
    const float* __restrict__ V,    // [seq_len × n_kv_heads × head_dim]
    float* __restrict__ out,        // [seq_len × n_heads × head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale
) {
    const int h = blockIdx.x;       // query head
    const int qpos = blockIdx.y;    // query position (0..seq_len-1)
    if (h >= n_heads || qpos >= seq_len) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float sdata[];
    float* scores = sdata;           // [qpos+1]
    float* ws = sdata + qpos + 1;   // [nthreads] workspace

    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    const float* q_head = Q + qpos * q_stride + h * head_dim;
    int ctx_len = qpos + 1;  // causal: attend to positions 0..qpos

    // Phase 1: QK dot products
    float lmax = -1e30f;
    for (int t = tid; t < ctx_len; t += nthreads) {
        const float* k_head = K + t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_head[d] * k_head[d];
        scores[t] = dot * scale;
        lmax = fmaxf(lmax, scores[t]);
    }

    ws[tid] = lmax;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid+s]); __syncthreads(); }
    float max_val = ws[0]; __syncthreads();

    float lsum = 0.0f;
    for (int t = tid; t < ctx_len; t += nthreads) { float e = expf(scores[t] - max_val); scores[t] = e; lsum += e; }
    ws[tid] = lsum; __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] += ws[tid+s]; __syncthreads(); }
    float sv = ws[0]; __syncthreads();
    for (int t = tid; t < ctx_len; t += nthreads) scores[t] /= sv;
    __syncthreads();

    // Phase 2: weighted V sum
    float* out_head = out + qpos * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            const float* v_head = V + t * kv_stride + kv_h * head_dim;
            val += scores[t] * v_head[d];
        }
        out_head[d] = val;
    }
}
