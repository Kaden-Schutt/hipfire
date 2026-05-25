// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_hfq4_kv(
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
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int bpb = 8 + head_dim / 2;  // 72 for head_dim=128
    const int bpp = n_kv_heads * bpb;

    float* scores = sdata;
    float* ws = sdata + seq_len;
    float* q_sh = ws + nthreads;  // Q preloaded in shared memory

    // Preload Q head into shared memory
    const float* q_head = q + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads)
        q_sh[d] = q_head[d];
    __syncthreads();

    // Phase 1: Q @ K^T — one 72-byte block per position, tight nibble loop
    float lmax = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* blk = k_cache + (size_t)t * bpp + kv_h * bpb;
        float sc = *(const float*)(blk);
        float zr = *(const float*)(blk + 4);
        float dot = 0.0f;
        // 64 bytes of nibbles = 128 elements, 2 per byte
        for (int j = 0; j < head_dim / 2; j++) {
            unsigned char pk = blk[8 + j];
            dot += q_sh[j * 2]     * (sc * (float)(pk & 0xF) + zr)
                 + q_sh[j * 2 + 1] * (sc * (float)(pk >> 4) + zr);
        }
        scores[t] = dot * scale_attn;
        lmax = fmaxf(lmax, scores[t]);
    }

    ws[tid] = lmax;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]); __syncthreads(); }
    float max_val = ws[0]; __syncthreads();

    float lsum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) { float e = expf(scores[t] - max_val); scores[t] = e; lsum += e; }
    ws[tid] = lsum; __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) { if (tid < s) ws[tid] += ws[tid + s]; __syncthreads(); }
    float sv = ws[0]; __syncthreads();
    for (int t = tid; t < seq_len; t += nthreads) scores[t] /= sv;
    __syncthreads();

    // Phase 2: weighted V sum — same 72-byte block dequant
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int byte_idx = d / 2;
        int is_high = d & 1;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* blk = v_cache + (size_t)t * bpp + kv_h * bpb;
            float sc = *(const float*)(blk);
            float zr = *(const float*)(blk + 4);
            unsigned char pk = blk[8 + byte_idx];
            val += scores[t] * (sc * (float)(is_high ? (pk >> 4) : (pk & 0xF)) + zr);
        }
        out_head[d] = val;
    }
}
