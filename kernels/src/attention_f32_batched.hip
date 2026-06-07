// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched causal attention with FP32 KV cache.
//
// Grid: [n_heads, batch_size, 1]. Each block computes one query head for one
// batch row. K/V caches are [max_seq, n_kv_heads, head_dim] FP32 and must
// already contain all positions referenced by positions[].
extern "C" __global__ void attention_f32_batched(
    const float* __restrict__ q,          // [B, n_heads, head_dim]
    const float* __restrict__ k_cache,    // [max_seq, n_kv_heads, head_dim]
    const float* __restrict__ v_cache,    // [max_seq, n_kv_heads, head_dim]
    float* __restrict__ out,              // [B, n_heads, head_dim]
    const int* __restrict__ positions,    // [B]
    const float* __restrict__ tree_bias,  // optional [B, block_cols] or nullptr
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;

    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[b] + 1);
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
        if (tree_mode && t >= block_start) {
            s += tree_bias[b * block_cols + (t - block_start)];
        }
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

// Batched causal FP32 attention specialized for GQA groups where four query
// heads share one KV head. This is intended for full-precision KLD prefill:
// it preserves the per-head FP32 dot-product and output accumulation order of
// attention_f32_batched, but computes four query heads in one block so K/V
// rows are loaded once for the group instead of once per query head.
//
// Grid: [n_kv_heads * ((n_heads / n_kv_heads) / 4), batch_size, 1].
// Shared memory layout uses max_ctx_len as the score stride:
//   scores    [4, max_ctx_len]
//   workspace [4, blockDim.x]
//   q_shared  [4, head_dim]
extern "C" __global__ void attention_f32_batched_gqa4(
    const float* __restrict__ q,          // [B, n_heads, head_dim]
    const float* __restrict__ k_cache,    // [max_seq, n_kv_heads, head_dim]
    const float* __restrict__ v_cache,    // [max_seq, n_kv_heads, head_dim]
    float* __restrict__ out,              // [B, n_heads, head_dim]
    const int* __restrict__ positions,    // [B]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_ctx_len,
    float scale_attn
) {
    const int group4 = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int kv_group = n_heads / n_kv_heads;
    const int groups_per_kv = kv_group / 4;
    const int kv_h = group4 / groups_per_kv;
    const int group_in_kv = group4 - kv_h * groups_per_kv;
    const int head_base = kv_h * kv_group + group_in_kv * 4;
    if (kv_h >= n_kv_heads || head_base + 3 >= n_heads) return;

    const int seq_len = positions[b] + 1;
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    extern __shared__ float sdata[];
    float* scores = sdata;
    float* workspace = scores + 4 * max_ctx_len;
    float* q_shared = workspace + 4 * nthreads;

    for (int h4 = 0; h4 < 4; ++h4) {
        const float* q_head = q + (long long)b * q_stride + (head_base + h4) * head_dim;
        float* q_dst = q_shared + h4 * head_dim;
        for (int d = tid; d < head_dim; d += nthreads) {
            q_dst[d] = q_head[d];
        }
    }
    __syncthreads();

    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
    float local_max2 = -1e30f;
    float local_max3 = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + (size_t)t * kv_stride + kv_h * head_dim;
        const float* q0 = q_shared + 0 * head_dim;
        const float* q1 = q_shared + 1 * head_dim;
        const float* q2 = q_shared + 2 * head_dim;
        const float* q3 = q_shared + 3 * head_dim;
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float kk = k_t[d];
            dot0 += q0[d] * kk;
            dot1 += q1[d] * kk;
            dot2 += q2[d] * kk;
            dot3 += q3[d] * kk;
        }
        float s0 = dot0 * scale_attn;
        float s1 = dot1 * scale_attn;
        float s2 = dot2 * scale_attn;
        float s3 = dot3 * scale_attn;
        scores[0 * max_ctx_len + t] = s0;
        scores[1 * max_ctx_len + t] = s1;
        scores[2 * max_ctx_len + t] = s2;
        scores[3 * max_ctx_len + t] = s3;
        local_max0 = fmaxf(local_max0, s0);
        local_max1 = fmaxf(local_max1, s1);
        local_max2 = fmaxf(local_max2, s2);
        local_max3 = fmaxf(local_max3, s3);
    }

    workspace[0 * nthreads + tid] = local_max0;
    workspace[1 * nthreads + tid] = local_max1;
    workspace[2 * nthreads + tid] = local_max2;
    workspace[3 * nthreads + tid] = local_max3;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            workspace[0 * nthreads + tid] =
                fmaxf(workspace[0 * nthreads + tid], workspace[0 * nthreads + tid + s]);
            workspace[1 * nthreads + tid] =
                fmaxf(workspace[1 * nthreads + tid], workspace[1 * nthreads + tid + s]);
            workspace[2 * nthreads + tid] =
                fmaxf(workspace[2 * nthreads + tid], workspace[2 * nthreads + tid + s]);
            workspace[3 * nthreads + tid] =
                fmaxf(workspace[3 * nthreads + tid], workspace[3 * nthreads + tid + s]);
        }
        __syncthreads();
    }
    const float max0 = workspace[0 * nthreads];
    const float max1 = workspace[1 * nthreads];
    const float max2 = workspace[2 * nthreads];
    const float max3 = workspace[3 * nthreads];
    __syncthreads();

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    float local_sum2 = 0.0f;
    float local_sum3 = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e0 = expf(scores[0 * max_ctx_len + t] - max0);
        float e1 = expf(scores[1 * max_ctx_len + t] - max1);
        float e2 = expf(scores[2 * max_ctx_len + t] - max2);
        float e3 = expf(scores[3 * max_ctx_len + t] - max3);
        scores[0 * max_ctx_len + t] = e0;
        scores[1 * max_ctx_len + t] = e1;
        scores[2 * max_ctx_len + t] = e2;
        scores[3 * max_ctx_len + t] = e3;
        local_sum0 += e0;
        local_sum1 += e1;
        local_sum2 += e2;
        local_sum3 += e3;
    }

    workspace[0 * nthreads + tid] = local_sum0;
    workspace[1 * nthreads + tid] = local_sum1;
    workspace[2 * nthreads + tid] = local_sum2;
    workspace[3 * nthreads + tid] = local_sum3;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            workspace[0 * nthreads + tid] += workspace[0 * nthreads + tid + s];
            workspace[1 * nthreads + tid] += workspace[1 * nthreads + tid + s];
            workspace[2 * nthreads + tid] += workspace[2 * nthreads + tid + s];
            workspace[3 * nthreads + tid] += workspace[3 * nthreads + tid + s];
        }
        __syncthreads();
    }
    const float sum0 = workspace[0 * nthreads];
    const float sum1 = workspace[1 * nthreads];
    const float sum2 = workspace[2 * nthreads];
    const float sum3 = workspace[3 * nthreads];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[0 * max_ctx_len + t] /= sum0;
        scores[1 * max_ctx_len + t] /= sum1;
        scores[2 * max_ctx_len + t] /= sum2;
        scores[3 * max_ctx_len + t] /= sum3;
    }
    __syncthreads();

    float* out0 = out + (long long)b * q_stride + (head_base + 0) * head_dim;
    float* out1 = out + (long long)b * q_stride + (head_base + 1) * head_dim;
    float* out2 = out + (long long)b * q_stride + (head_base + 2) * head_dim;
    float* out3 = out + (long long)b * q_stride + (head_base + 3) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val0 = 0.0f;
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float vv = v_cache[(size_t)t * kv_stride + kv_h * head_dim + d];
            val0 += scores[0 * max_ctx_len + t] * vv;
            val1 += scores[1 * max_ctx_len + t] * vv;
            val2 += scores[2 * max_ctx_len + t] * vv;
            val3 += scores[3 * max_ctx_len + t] * vv;
        }
        out0[d] = val0;
        out1[d] = val1;
        out2[d] = val2;
        out3[d] = val3;
    }
}
