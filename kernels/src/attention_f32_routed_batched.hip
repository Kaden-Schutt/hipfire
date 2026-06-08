// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Routed batched causal attention with FP32 KV cache.
//
// Grid: [n_heads, batch_size, 1]. Each block computes one query head for one
// routed prefix row. K/V caches are selected by row_session_indices[b] and
// layer_index from session-major pointer tables:
//
//   k_cache = k_ptrs[row_session * ptr_layer_stride + layer_index]
//   v_cache = v_ptrs[row_session * ptr_layer_stride + layer_index]
//
// This is the session-batched sibling of attention_f32_batched for AR prefill.
extern "C" __global__ void attention_f32_routed_batched(
    const float* __restrict__ q,                    // [B, n_heads, head_dim]
    const uint64_t* __restrict__ k_ptrs,            // [n_sessions * ptr_layer_stride]
    const uint64_t* __restrict__ v_ptrs,            // [n_sessions * ptr_layer_stride]
    float* __restrict__ out,                        // [B, n_heads, head_dim]
    const int* __restrict__ row_session_indices,    // [B]
    const int* __restrict__ positions,              // [B]
    int ptr_layer_stride,
    int layer_index,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;

    const int session_idx = row_session_indices[b];
    const float* k_cache = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(k_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index])
    );
    const float* v_cache = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(v_ptrs[(size_t)session_idx * ptr_layer_stride + layer_index])
    );

    const int seq_len = positions[b] + 1;
    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int q_stride = n_heads * head_dim;

    const float* q_head = q + (long long)b * q_stride + h * head_dim;
    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    for (int d = tid; d < head_dim; d += nthreads) {
        q_shared[d] = q_head[d];
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_shared[d] * k_t[d];
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
    const float max_val = workspace[0];
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
    const float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    float* out_head = out + (long long)b * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            val += scores[t] * v_cache[(size_t)t * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        out_head[d] = val;
    }
}
