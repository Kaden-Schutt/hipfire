// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_int8c_kv(
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

    const int bph = 8 + head_dim;  // 8-byte header (scale + pad) + data
    const int bpp = n_kv_heads * bph;

    float* scores = sdata;
    float* ws = sdata + seq_len;
    float* q_sh = ws + nthreads;

    // Preload Q into shared memory
    const float* q_head = q + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads)
        q_sh[d] = q_head[d];
    __syncthreads();

    // Phase 1: QK with deferred scale, vector loads, 4-wide accumulation
    float lmax = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* blk = k_cache + (size_t)t * bpp + kv_h * bph;
        float sc = *(const float*)(blk);
        const signed char* kd = (const signed char*)(blk + 8);

        // 4-wide unrolled dot product with deferred scale
        float acc = 0.0f;
        for (int j = 0; j < head_dim; j += 4) {
            acc += q_sh[j]   * (float)kd[j]
                 + q_sh[j+1] * (float)kd[j+1]
                 + q_sh[j+2] * (float)kd[j+2]
                 + q_sh[j+3] * (float)kd[j+3];
        }
        scores[t] = sc * acc * scale_attn;
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

    // Phase 2: weighted V sum — precompute score*scale per position
    // This avoids loading scale once per (position × dimension) — loads it once per position
    float* sv_cache = ws;  // reuse workspace for score*scale products
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* blk = v_cache + (size_t)t * bpp + kv_h * bph;
        sv_cache[t] = scores[t] * *(const float*)(blk);
    }
    __syncthreads();

    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* blk = v_cache + (size_t)t * bpp + kv_h * bph;
            val += sv_cache[t] * (float)((signed char)blk[8 + d]);
        }
        out_head[d] = val;
    }
}
