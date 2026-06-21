// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention tile for f16 K + Q8_0 V — the v1 KVarN read path.
// KVarN history is materialized into a token-major f16 shadow K cache
// (`kvarn_build_kcache`); V stays Q8_0 (reuses the asym4/q8 V layout). This is a
// clone of attention_flash_q8_0_tile_batched with ONLY Phase A (Q·K) changed:
// K is read directly as f16 from a contiguous `[max_seq × kv_dim]` buffer
// instead of dequantizing Q8 blocks. Partials layout / reduce kernel / V path
// are byte-identical to the Q8 tile, so it shares
// attention_flash_asym_reduce_batched and launch_asym_flash_batched.
//
// cos_theta/sin_theta accepted for ABI compatibility but ignored (K unrotated).
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_f16k_q8v_tile_batched(
    const float* __restrict__ q,               // [sub_batch × n_heads × head_dim]
    const _Float16* __restrict__ k_cache,      // [max_seq × kv_dim] f16 token-major
    const unsigned char* __restrict__ v_cache, // Q8_0
    float* __restrict__ partials,
    const int* __restrict__ positions,         // [total_batch]
    const float* __restrict__ /*cos_theta*/,   // unused (K unrotated)
    const float* __restrict__ /*sin_theta*/,   // unused
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
    int block_cols                             // ignored when tree_bias == nullptr
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

    // V is Q8_0: head_dim/32 blocks/head, 34 bytes/block.
    const int blocks_per_head = head_dim / 32;
    const int total_blocks    = n_kv_heads * blocks_per_head;
    const int kv_head_blk     = kv_h * blocks_per_head;
    const int v_row_stride    = total_blocks * 34;
    // K is f16 token-major: kv_dim elems/token, kv-head offset kv_h*head_dim.
    const int kv_dim          = n_kv_heads * head_dim;
    const int k_head_off      = kv_h * head_dim;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position: 8 consecutive dims per thread.
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int d0 = tid * 8;
    float mq[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) mq[i] = q_head[d0 + i];

    // Phase A: Q·K — K read directly as f16 (8 dims per thread).
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const _Float16* kt = k_cache + (size_t)t * kv_dim + k_head_off;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            partial += mq[i] * (float)kt[d0 + i];
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= block_start)
                s += tree_bias[global_bid * block_cols + (t - block_start)];
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

    // Phase D: V-weighted accumulation (Q8_0) — identical to the Q8 tile.
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int d = d0 + i;
            const int bi = d / 32;
            const int bj = d % 32;
            const unsigned char* vbh = vb + (kv_head_blk + bi) * 34;
            float vs = (float)*((const _Float16*)vbh);
            out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
        }
    }

    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + tile_id * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++)
        p[2 + d0 + i] = out_vec[i];
}
