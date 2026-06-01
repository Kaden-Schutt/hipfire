// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile — fwht3 KV (K FWHT-rotated 3-bit; V at Q8_0
// OR lloyd3 FWHT-rotated centroid, gated by v_mode).
// Single-pass 256-element signed-FWHT on Q to match the K-write rotation.
// Partials layout: [sub_batch × n_heads × max_tiles × (2+head_dim)].
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
// LDS: tile_size * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Tree-attention mode contract matches attention_flash_asym3_tile_batched.
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht3_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols,
    int v_mode          // 8 = legacy Q8_0 V; 3 = lloyd3 FWHT-rotated centroid V
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols) : (positions[global_bid] + 1);
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0 = tid * 8;
    float mq0 = q_head[d0 + 0], mq1 = q_head[d0 + 1];
    float mq2 = q_head[d0 + 2], mq3 = q_head[d0 + 3];
    float mq4 = q_head[d0 + 4], mq5 = q_head[d0 + 5];
    float mq6 = q_head[d0 + 6], mq7 = q_head[d0 + 7];

    fwht_shfl_forward_256(mq0, mq1, mq2, mq3, mq4, mq5, mq6, mq7, signs1, signs2, tid);

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * k_bytes_per_pos + k_head_off;
        float cnorm = *(const float*)kb;
        const unsigned char* base = kb + 4 + tid * 3;
        unsigned int packed = (unsigned int)base[0]
                            | ((unsigned int)base[1] << 8)
                            | ((unsigned int)base[2] << 16);
        float partial = mq0 * (cnorm * TURBO_C3_256[ packed        & 7])
                      + mq1 * (cnorm * TURBO_C3_256[(packed >> 3)  & 7])
                      + mq2 * (cnorm * TURBO_C3_256[(packed >> 6)  & 7])
                      + mq3 * (cnorm * TURBO_C3_256[(packed >> 9)  & 7])
                      + mq4 * (cnorm * TURBO_C3_256[(packed >> 12) & 7])
                      + mq5 * (cnorm * TURBO_C3_256[(packed >> 15) & 7])
                      + mq6 * (cnorm * TURBO_C3_256[(packed >> 18) & 7])
                      + mq7 * (cnorm * TURBO_C3_256[(packed >> 21) & 7]);
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= block_start) {
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            }
            scores[t_local] = s;
        }
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    if (v_mode == 8) {
        // Legacy Q8_0 V (normal space) — unchanged behavior.
        const int v_blocks_per_head = head_dim / 32;
        const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
        const int v_kv_head_blk     = kv_h * v_blocks_per_head;
        const int v_row_stride      = v_total_blocks * 34;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const int v_blk_off = (v_kv_head_blk + bi) * 34;
                const unsigned char* vbh = vb + v_blk_off;
                float vs = (float)*((const _Float16*)vbh);
                out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
            }
        }
    } else {
        // lloyd2/3/4: 256-wide FWHT-rotated centroid V. v_mode = bit width.
        // Layout: 4B cnorm + (tid * v_mode) bytes packed (8 indices, LSB-first, i*v_mode shift).
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            float cnorm = *(const float*)vb;
            const unsigned char* base = vb + 4 + tid * v_mode;   // v_mode bytes/thread
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float c;
                if (v_mode == 2)      c = TURBO_C2_256[(packed >> (i * 2)) & 3];
                else if (v_mode == 3) c = TURBO_C3_256[(packed >> (i * 3)) & 7];
                else                  c = TURBO_C4_256[(packed >> (i * 4)) & 15];
                out_vec[i] += w * (cnorm * c);
            }
        }
        // V accumulated in ROTATED space (out_vec ≈ H·Σw·v). The inverse FWHT
        // is deferred to the dedicated lloyd reduce kernel, which applies it
        // once after the cross-tile combine (inverse is linear, so it commutes
        // with the weighted-sum + scalar normalize the reduce performs).
    }

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        p[2 + d0 + i] = out_vec[i];
    }
}
