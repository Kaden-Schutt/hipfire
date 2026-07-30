// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Experimental Q8_0 prefill attention tile that processes four adjacent
// queries per workgroup. K/V values are dequantized once and reused by all
// four queries, reducing the dominant long-prefill KV traffic versus the
// one-query-per-workgroup scalar tile.
//
// This kernel intentionally keeps the existing partials ABI so the established
// attention_flash_asym_reduce_batched kernel performs the cross-tile online
// softmax reduction unchanged.
//
// Constraints enforced by the gfx12-only host opt-in route:
//   - batch chunks are multiples of BLOCK_M
//   - causal attention only (no tree bias or sliding window)
//   - head_dim is 128 or 256
//
// Grid: [n_heads, ceil(sub_batch/BLOCK_M), max_tiles]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>

#define BLOCK_M 4
#define TILE_SIZE 128

extern "C" __launch_bounds__(32, 4)
__global__ void attention_flash_q8_0_m4_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ /*cos_theta*/,
    const float* __restrict__ /*sin_theta*/,
    const float* __restrict__ /*tree_bias*/,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int /*block_start*/,
    int /*block_cols*/,
    int /*v_mode_bits*/,
    int /*window*/
) {
    const int h = blockIdx.x;
    const int m_start = blockIdx.y * BLOCK_M;
    const int tile_id = blockIdx.z;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int tile_start = tile_id * tile_size;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int dpt = head_dim / 32;
    const int d0 = tid * dpt;

    int seq_len[BLOCK_M];
    int tile_len[BLOCK_M];
    bool any_work = false;
    int max_tile_len = 0;
#pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        const int global_bid = batch_offset + m_start + m;
        seq_len[m] = positions[global_bid] + 1;
        int len = seq_len[m] - tile_start;
        len = max(0, min(min(tile_size, max_seq - tile_start), len));
        tile_len[m] = len;
        any_work |= len > 0;
        max_tile_len = max(max_tile_len, len);
    }
    if (!any_work) return;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    const int kv_head_blk = kv_h * blocks_per_head;
    const int row_stride = total_blocks * 34;

    float mq[BLOCK_M][8];
#pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        const float* q_head =
            q + (size_t)(m_start + m) * q_dim + h * head_dim;
        for (int i = 0; i < dpt; i++) mq[m][i] = q_head[d0 + i];
    }

    extern __shared__ float sdata[];
    float (*scores)[TILE_SIZE] = (float (*)[TILE_SIZE])sdata;

    for (int t_local = 0; t_local < max_tile_len; t_local++) {
        const int t = tile_start + t_local;
        float dot[BLOCK_M] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < dpt; i++) {
            const int d = d0 + i;
            const int bi = d / 32;
            const int bj = d % 32;
            const unsigned char* kb =
                k_cache + (size_t)t * row_stride + (kv_head_blk + bi) * 34;
            const float ks = (float)*((const _Float16*)kb);
            const float kval = ks * (float)((signed char)kb[2 + bj]);
#pragma unroll
            for (int m = 0; m < BLOCK_M; m++) {
                dot[m] += mq[m][i] * kval;
            }
        }
#pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            for (int off = 16; off > 0; off >>= 1)
                dot[m] += __shfl_xor(dot[m], off);
            if (tid == 0 && t_local < tile_len[m])
                scores[m][t_local] = dot[m] * scale_attn;
        }
    }
    __syncthreads();

    float tile_max[BLOCK_M];
    float tile_sum[BLOCK_M];
#pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        float local_max = -1e30f;
        for (int i = tid; i < tile_len[m]; i += 32)
            local_max = fmaxf(local_max, scores[m][i]);
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor(local_max, off));
        tile_max[m] = local_max;

        float local_sum = 0.0f;
        for (int i = tid; i < tile_len[m]; i += 32) {
            const float e = expf(scores[m][i] - local_max);
            scores[m][i] = e;
            local_sum += e;
        }
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_xor(local_sum, off);
        tile_sum[m] = local_sum;
    }
    __syncthreads();

    float out_vec[BLOCK_M][8] = {};
    for (int t_local = 0; t_local < max_tile_len; t_local++) {
        const int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * row_stride;
        for (int i = 0; i < dpt; i++) {
            const int d = d0 + i;
            const int bi = d / 32;
            const int bj = d % 32;
            const unsigned char* vbh = vb + (kv_head_blk + bi) * 34;
            const float vs = (float)*((const _Float16*)vbh);
            const float vval = vs * (float)((signed char)vbh[2 + bj]);
#pragma unroll
            for (int m = 0; m < BLOCK_M; m++) {
                if (t_local < tile_len[m])
                    out_vec[m][i] += scores[m][t_local] * vval;
            }
        }
    }

    const int stride = 2 + head_dim;
#pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        if (tile_len[m] == 0) continue;
        float* p = partials
            + ((long long)(m_start + m) * n_heads + h) * max_tiles * stride
            + tile_id * stride;
        if (tid == 0) {
            p[0] = tile_max[m];
            p[1] = tile_sum[m];
        }
        for (int i = 0; i < dpt; i++)
            p[2 + d0 + i] = out_vec[m][i];
    }
}
