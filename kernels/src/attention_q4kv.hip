// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void attention_q4kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache_q4,
    const unsigned char* __restrict__ v_cache_q4,
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

    const float* q_head = q + h * head_dim;
    int bytes_per_head = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;

    // Phase 1: Q @ K^T with on-the-fly dequant
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* k_packed = k_cache_q4 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
        float k_scale = *(const float*)(k_packed);
        float k_zero  = *(const float*)(k_packed + 4);
        const unsigned char* k_nib = k_packed + 8;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d += 2) {
            unsigned char byte_val = k_nib[d / 2];
            float k0 = k_scale * (float)(byte_val & 0xF) + k_zero;
            float k1 = k_scale * (float)(byte_val >> 4) + k_zero;
            dot += q_head[d] * k0 + q_head[d + 1] * k1;
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of V with on-the-fly dequant
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int d_byte = d / 2;
        int d_nib_shift = (d & 1) * 4;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* v_packed = v_cache_q4 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
            float v_scale = *(const float*)(v_packed);
            float v_zero  = *(const float*)(v_packed + 4);
            unsigned char byte_val = v_packed[8 + d_byte];
            float v_val = v_scale * (float)((byte_val >> d_nib_shift) & 0xF) + v_zero;
            val += scores[t] * v_val;
        }
        out_head[d] = val;
    }
}
