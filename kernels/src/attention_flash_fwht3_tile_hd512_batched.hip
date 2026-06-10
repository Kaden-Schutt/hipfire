// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile — fwht3 KV, head_dim=512 (Gemma 4 full-attention layers).
//
// Combines the hd=512 per-thread 16-dim Q/K/V handling from
// attention_flash_fwht3_tile_hd512.hip with the batched scheme from
// attention_flash_fwht3_tile_batched.hip: blockIdx.z is the batch slot,
// positions[] holds per-batch absolute positions, and partials are laid out
// as [sub_batch × n_heads × max_tiles × (2 + head_dim)].
//
// K byte geometry (matches single-token variant):
//   bytes_per_head = 4 + (512*3)/8 = 196
//   lane tid: 6 bytes at offset 4 + tid*6 (contiguous-per-lane)
//   packed0 = idx[0..7]  (24 bits); packed1 = idx[8..15] (24 bits)
//
// V side: Q8_0 normal space (v_mode=8). Reuses attention_flash_q8_0_reduce
// unchanged (n_halves=4, stride 514).
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
// LDS: tile_size * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_fwht3_tile_hd512_batched(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const float* __restrict__ tree_bias,       // optional [total_batch × block_cols]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                           // ignored when tree_bias == nullptr
    int block_cols,                            // ignored when tree_bias == nullptr
    int window_size,                           // 0 = full causal (Gemma 4 full layers)
    int cache_capacity,                        // 0 = no ring wrap
    int v_mode                                 // 8 = Q8_0 V; 3 = lloyd3 (future)
) {
    if (head_dim != 512) return;

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
    const bool sliding = (window_size > 0) && !tree_mode;
    const int t_lo = sliding ? max(0, seq_len - window_size) : 0;

    // Out-of-window early exit: zero this tile's partials.
    if (sliding && tile_end <= t_lo) {
        float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                            + tile_id * (2 + head_dim);
        if (tid == 0) { p[0] = -1e30f; p[1] = 0.0f; }
        const int d0z = tid * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            p[2 + d0z + i] = 0.0f;
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    // K geometry: 4 + 192 = 196 bytes per head.
    const int k_bytes_per_head = 4 + (head_dim * 3) / 8;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    // V geometry (Q8_0). 16 blocks/head at hd=512.
    const int v_blocks_per_head = head_dim / 32;
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q: 16 values per thread, contiguous at d0 = tid*16.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0 = tid * 16;
    float mq0  = q_head[d0 +  0], mq1  = q_head[d0 +  1];
    float mq2  = q_head[d0 +  2], mq3  = q_head[d0 +  3];
    float mq4  = q_head[d0 +  4], mq5  = q_head[d0 +  5];
    float mq6  = q_head[d0 +  6], mq7  = q_head[d0 +  7];
    float mq8  = q_head[d0 +  8], mq9  = q_head[d0 +  9];
    float mq10 = q_head[d0 + 10], mq11 = q_head[d0 + 11];
    float mq12 = q_head[d0 + 12], mq13 = q_head[d0 + 13];
    float mq14 = q_head[d0 + 14], mq15 = q_head[d0 + 15];

    // FWHT-512 forward transform.
    fwht_shfl_forward_512(
        mq0,  mq1,  mq2,  mq3,  mq4,  mq5,  mq6,  mq7,
        mq8,  mq9,  mq10, mq11, mq12, mq13, mq14, mq15,
        signs1, signs2, tid);

    // Phase A: Q·K (3-bit K, two 24-bit packed words per lane).
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const bool out_of_window = sliding && (t < t_lo);
        float partial = 0.0f;
        if (!out_of_window) {
            const int k_slot = (cache_capacity > 0) ? (t % cache_capacity) : t;
            const unsigned char* kb = k_cache + (size_t)k_slot * k_bytes_per_pos + k_head_off;
            float cnorm = *(const float*)kb;
            // Lane tid: 6 bytes at offset 4 + tid*6 (contiguous-per-lane layout).
            const unsigned char* base = kb + 4 + tid * 6;
            unsigned int packed0 = (unsigned int)base[0]
                                 | ((unsigned int)base[1] << 8)
                                 | ((unsigned int)base[2] << 16);
            unsigned int packed1 = (unsigned int)base[3]
                                 | ((unsigned int)base[4] << 8)
                                 | ((unsigned int)base[5] << 16);
            partial  = mq0  * (cnorm * TURBO_C3_512[ packed0        & 7])
                     + mq1  * (cnorm * TURBO_C3_512[(packed0 >>  3) & 7])
                     + mq2  * (cnorm * TURBO_C3_512[(packed0 >>  6) & 7])
                     + mq3  * (cnorm * TURBO_C3_512[(packed0 >>  9) & 7])
                     + mq4  * (cnorm * TURBO_C3_512[(packed0 >> 12) & 7])
                     + mq5  * (cnorm * TURBO_C3_512[(packed0 >> 15) & 7])
                     + mq6  * (cnorm * TURBO_C3_512[(packed0 >> 18) & 7])
                     + mq7  * (cnorm * TURBO_C3_512[(packed0 >> 21) & 7])
                     + mq8  * (cnorm * TURBO_C3_512[ packed1        & 7])
                     + mq9  * (cnorm * TURBO_C3_512[(packed1 >>  3) & 7])
                     + mq10 * (cnorm * TURBO_C3_512[(packed1 >>  6) & 7])
                     + mq11 * (cnorm * TURBO_C3_512[(packed1 >>  9) & 7])
                     + mq12 * (cnorm * TURBO_C3_512[(packed1 >> 12) & 7])
                     + mq13 * (cnorm * TURBO_C3_512[(packed1 >> 15) & 7])
                     + mq14 * (cnorm * TURBO_C3_512[(packed1 >> 18) & 7])
                     + mq15 * (cnorm * TURBO_C3_512[(packed1 >> 21) & 7]);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = out_of_window ? -1e30f : (partial * scale_attn);
            if (tree_mode && t >= block_start) {
                s += tree_bias[global_bid * block_cols + (t - block_start)];
            }
            scores[t_local] = s;
        }
    }
    __syncthreads();

    // Phase B: tile max
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // Phase C: exp + sum
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

    // Phase D: V — 16 output dims per thread (contiguous d0..d0+15).
    float out_vec[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    if (v_mode == 8) {
        // Q8_0 V (normal space) — same logic as asym3_hd512_batched but with
        // contiguous d0 = tid*16 map instead of the two-chunk d0_lo/d0_hi split.
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const int v_slot = (cache_capacity > 0) ? (t % cache_capacity) : t;
            const unsigned char* vb = v_cache + (size_t)v_slot * v_row_stride;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d & 31;
                const unsigned char* vbh = vb + (v_kv_head_blk + bi) * 34;
                float vs = (float)*((const _Float16*)vbh);
                out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
            }
        }
    }
    // v_mode==3 (lloyd3 rotated-V) deferred per spec §2.4.

    // Write partials — batched layout: [sub_batch × n_heads × max_tiles × stride].
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        p[2 + d0 + i] = out_vec[i];
    }
}
